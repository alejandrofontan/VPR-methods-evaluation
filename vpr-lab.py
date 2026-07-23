import os
import parser
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import vpr_models
import yaml
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


def extract_descriptors_worker(rank, world_size, args, rotation_angle, yaml_data, all_descriptors):
    device = f"cuda:{rank}" if args.device == "cuda" else args.device

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(device)

    test_ds = build_test_dataset(yaml_data, args, rotation_angle)

    database_shard = np.array_split(np.arange(test_ds.num_database), world_size)[rank].tolist()
    queries_shard = np.array_split(
        np.arange(test_ds.num_database, test_ds.num_database + test_ds.num_queries), world_size
    )[rank].tolist()

    with torch.inference_mode():
        database_dataloader = DataLoader(
            dataset=Subset(test_ds, database_shard), num_workers=args.num_workers, batch_size=args.batch_size
        )
        for images, indices in tqdm(
            database_dataloader,
            desc=f"{SCRIPT_LABEL}[gpu {rank}] Extracting database descriptors (rotation {rotation_angle} deg)",
        ):
            descriptors = model(images.to(device)).cpu()
            all_descriptors[indices] = descriptors

        queries_dataloader = DataLoader(dataset=Subset(test_ds, queries_shard), num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(
            queries_dataloader,
            desc=f"{SCRIPT_LABEL}[gpu {rank}] Extracting query descriptors (rotation {rotation_angle} deg)",
        ):
            descriptors = model(images.to(device)).cpu()
            all_descriptors[indices] = descriptors


def print_gpu_info(args, world_size):
    if args.device != "cuda" or torch.cuda.device_count() == 0:
        print(f"{SCRIPT_LABEL}Using CPU")
        return

    print(f"{SCRIPT_LABEL}Using {world_size} GPU(s):")
    for rank in range(world_size):
        name = torch.cuda.get_device_name(rank)
        free_bytes, total_bytes = torch.cuda.mem_get_info(rank)
        print(f"{SCRIPT_LABEL}    - cuda:{rank} {name} ({free_bytes / 1024**3:.1f} / {total_bytes / 1024**3:.1f} GB free)")


def run_vpr(args, rotation_angle, yaml_data):
    log_dir = Path(yaml_data["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    sizing_ds = build_test_dataset(yaml_data, args, rotation_angle)
    num_database = sizing_ds.num_database
    total_images = len(sizing_ds)
    del sizing_ds

    all_descriptors = torch.zeros((total_images, args.descriptors_dimension), dtype=torch.float32)
    all_descriptors.share_memory_()

    world_size = resolve_world_size(args)
    print_gpu_info(args, world_size)

    if world_size == 1:
        extract_descriptors_worker(0, 1, args, rotation_angle, yaml_data, all_descriptors)
    else:
        mp.spawn(
            extract_descriptors_worker,
            args=(world_size, args, rotation_angle, yaml_data, all_descriptors),
            nprocs=world_size,
            join=True,
        )

    all_descriptors = all_descriptors.numpy()
    queries_descriptors = all_descriptors[num_database:]
    database_descriptors = all_descriptors[:num_database]

    # Compute similarity matrix
    distance_matrix = faiss.pairwise_distances(database_descriptors, queries_descriptors)
    np.save(os.path.join(log_dir, f"D_{rotation_angle}.npy"), distance_matrix)  # saves binary numpy file

    if args.verbose:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Distance")
        plt.xlabel("Query Descriptors")
        plt.ylabel("Database Descriptors")
        plt.title("Pairwise Distance Matrix")

        plt.show()

    return distance_matrix

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

    D_0 = run_vpr(args, 0, yaml_data)
    if args.rot_360:
        D_90 = run_vpr(args, 90, yaml_data)
        D_180 = run_vpr(args, 180, yaml_data)
        D_270 = run_vpr(args, 270, yaml_data)
        D = np.minimum.reduce([D_0, D_90, D_180, D_270])
    else:
        D = D_0
    np.save(os.path.join(yaml_data["log_dir"], f"D.npy"), D)
