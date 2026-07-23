import os
import parser
import shutil
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import vpr_models
import yaml
from PIL import Image
from rotation_functions import rotate_and_save_images
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def resolve_world_size(args) -> int:
    if args.device != "cuda":
        return 1
    available = torch.cuda.device_count()
    if available == 0:
        return 1
    if args.gpu is None:
        return available
    return min(args.gpu, available)


def resolve_num_workers(args, world_size) -> int:
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        available_cpus = os.cpu_count() or 1

    per_rank_cpus = max(1, available_cpus // world_size)
    return max(1, min(args.num_workers, per_rank_cpus))


def build_test_dataset(yaml_data, args, rotation_angle):
    database_rgb_csv_path = Path(yaml_data["rgb_list_db"])
    queries_rgb_csv_path = Path(yaml_data["rgb_list_q"])

    database_root = database_rgb_csv_path.parent
    query_root = queries_rgb_csv_path.parent

    db_database_paths_raw = pd.read_csv(database_rgb_csv_path)["path_rgb_0"]
    db_queries_paths_raw = pd.read_csv(queries_rgb_csv_path)["path_rgb_0"]

    database_image_list = [f"{database_root / p}" for p in db_database_paths_raw]
    queries_image_list = [f"{query_root / p}" for p in db_queries_paths_raw]

    return TestDataset(
        database_image_list=database_image_list,
        queries_image_list=queries_image_list,
        rotation=rotation_angle,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )


def images_share_resolution(dataset, indices) -> bool:
    sizes = set()
    for idx in indices:
        with Image.open(dataset.images_paths[idx]) as img:
            sizes.add(img.size)
        if len(sizes) > 1:
            return False
    return True


def estimate_batch_size(args, device, model, dataset, indices, safety_margin=0.7) -> int:
    if not str(device).startswith("cuda") or len(indices) == 0:
        return args.batch_size

    if args.image_size is None and not images_share_resolution(dataset, indices):
        return 1

    sample_image, _ = dataset[indices[0]]

    torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    with torch.inference_mode():
        model(sample_image.unsqueeze(0).to(device))
    torch.cuda.synchronize(device)
    per_image_bytes = torch.cuda.max_memory_allocated(device) - baseline_bytes

    if per_image_bytes <= 0:
        return min(args.batch_size, len(indices))

    free_bytes, _ = torch.cuda.mem_get_info(device)
    extra_images = int((free_bytes * safety_margin) // per_image_bytes)

    return max(1, min(1 + extra_images, len(indices)))


def run_extraction_loop(model, device, dataloader, all_descriptors, desc):
    data_time = 0.0
    compute_time = 0.0
    t0 = time.time()
    for images, indices in tqdm(dataloader, desc=desc):
        t1 = time.time()
        data_time += t1 - t0
        descriptors = model(images.to(device)).cpu()
        all_descriptors[indices] = descriptors
        t0 = time.time()
        compute_time += t0 - t1
    print(f"{SCRIPT_LABEL}    data_time = {data_time:.1f}s, compute_time = {compute_time:.1f}s")


def extract_descriptors_worker(rank, world_size, args, rotations, yaml_data, all_descriptors_by_rotation):
    device = f"cuda:{rank}" if args.device == "cuda" else args.device

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(device)

    num_workers = resolve_num_workers(args, world_size)
    print(f"{SCRIPT_LABEL}[gpu {rank}] num_workers = {num_workers}")

    for rotation_angle in rotations:
        all_descriptors = all_descriptors_by_rotation[rotation_angle]
        test_ds = build_test_dataset(yaml_data, args, rotation_angle)

        database_shard = np.array_split(np.arange(test_ds.num_database), world_size)[rank].tolist()
        queries_shard = np.array_split(
            np.arange(test_ds.num_database, test_ds.num_database + test_ds.num_queries), world_size
        )[rank].tolist()

        with torch.inference_mode():
            database_batch_size = estimate_batch_size(args, device, model, test_ds, database_shard)
            print(f"{SCRIPT_LABEL}[gpu {rank}] database batch_size = {database_batch_size} ({len(database_shard)} images)")
            database_dataloader = DataLoader(
                dataset=Subset(test_ds, database_shard), num_workers=num_workers, batch_size=database_batch_size
            )
            run_extraction_loop(
                model, device, database_dataloader, all_descriptors,
                desc=f"{SCRIPT_LABEL}[gpu {rank}] Extracting database descriptors (rotation {rotation_angle} deg)",
            )

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            queries_batch_size = estimate_batch_size(args, device, model, test_ds, queries_shard)
            print(f"{SCRIPT_LABEL}[gpu {rank}] queries batch_size = {queries_batch_size} ({len(queries_shard)} images)")
            queries_dataloader = DataLoader(
                dataset=Subset(test_ds, queries_shard), num_workers=num_workers, batch_size=queries_batch_size
            )
            run_extraction_loop(
                model, device, queries_dataloader, all_descriptors,
                desc=f"{SCRIPT_LABEL}[gpu {rank}] Extracting query descriptors (rotation {rotation_angle} deg)",
            )

            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()


def print_gpu_info(args, world_size):
    if args.device != "cuda" or torch.cuda.device_count() == 0:
        print(f"{SCRIPT_LABEL}Using CPU")
        return

    print(f"{SCRIPT_LABEL}Using {world_size} GPU(s):")
    for rank in range(world_size):
        name = torch.cuda.get_device_name(rank)
        free_bytes, total_bytes = torch.cuda.mem_get_info(rank)
        print(f"{SCRIPT_LABEL}    - cuda:{rank} {name} ({free_bytes / 1024**3:.1f} / {total_bytes / 1024**3:.1f} GB free)")


def run_vpr(args, rotations, yaml_data):
    log_dir = Path(yaml_data["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    # Rotation only changes the transform, not the dataset size, so any rotation is fine for sizing.
    sizing_ds = build_test_dataset(yaml_data, args, rotations[0])
    num_database = sizing_ds.num_database
    total_images = len(sizing_ds)
    del sizing_ds

    all_descriptors_by_rotation = {}
    for rotation_angle in rotations:
        t = torch.zeros((total_images, args.descriptors_dimension), dtype=torch.float32)
        t.share_memory_()
        all_descriptors_by_rotation[rotation_angle] = t

    world_size = resolve_world_size(args)
    print_gpu_info(args, world_size)

    if world_size == 1:
        extract_descriptors_worker(0, 1, args, rotations, yaml_data, all_descriptors_by_rotation)
    else:
        mp.spawn(
            extract_descriptors_worker,
            args=(world_size, args, rotations, yaml_data, all_descriptors_by_rotation),
            nprocs=world_size,
            join=True,
        )

    distance_matrices = {}
    for rotation_angle in rotations:
        all_descriptors = all_descriptors_by_rotation[rotation_angle].numpy()
        queries_descriptors = all_descriptors[num_database:]
        database_descriptors = all_descriptors[:num_database]

        distance_matrix = faiss.pairwise_distances(database_descriptors, queries_descriptors)
        np.save(os.path.join(log_dir, f"D_{rotation_angle}.npy"), distance_matrix)  # saves binary numpy file
        distance_matrices[rotation_angle] = distance_matrix

        if args.verbose:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(8, 6))
            plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
            plt.colorbar(label="Distance")
            plt.xlabel("Query Descriptors")
            plt.ylabel("Database Descriptors")
            plt.title(f"Pairwise Distance Matrix (rotation {rotation_angle} deg)")

            plt.show()

    return distance_matrices

if __name__ == "__main__":

    args = parser.parse_arguments()
    exp_yaml = args.exp_yaml

    if os.path.exists(exp_yaml):
        with open(exp_yaml, "r") as stream:
            yaml_data = yaml.safe_load(stream)
    else:
        yaml_data = {
            "rgb_list_db": args.rgb_csv_db,
            "rgb_list_q": args.rgb_csv_q,
            "log_dir": args.log_dir,
        }
    print(f"\n{SCRIPT_LABEL}Running VPR evaluation")
    print(f"    - Method:   {args.method}")
    print(f"    - Rotation: {args.rot_360}")
    print(f"    - Database: {yaml_data['rgb_list_db']}")
    print(f"    - Queries:  {yaml_data['rgb_list_q']}")
    print(f"    - Log dir:  {yaml_data['log_dir']}")

    rotations = [0, 90, 180, 270] if args.rot_360 else [0]
    start_time = time.time()
    distance_matrices = run_vpr(args, rotations, yaml_data)
    elapsed = time.time() - start_time

    D = np.minimum.reduce([distance_matrices[r] for r in rotations])
    np.save(os.path.join(yaml_data["log_dir"], f"D.npy"), D)

    print(f"{SCRIPT_LABEL}Total time: {elapsed:.2f}s")
