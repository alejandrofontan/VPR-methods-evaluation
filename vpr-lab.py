import os
import parser
import shutil
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import vpr_models
import yaml
from rotation_functions import rotate_and_save_images
from test_dataset import TestDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from tqdm import tqdm


def run_vpr(args, rotation_angle, yaml_data):

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    database_rgb_csv_path = Path(yaml_data["rgb_list_db"])
    queries_rgb_csv_path = Path(yaml_data["rgb_list_q"])

    database_root = database_rgb_csv_path.parent
    query_root = queries_rgb_csv_path.parent

    log_dir = Path(yaml_data["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    db_database_paths_raw = pd.read_csv(database_rgb_csv_path)["path_rgb_0"]
    db_queries_paths_raw = pd.read_csv(queries_rgb_csv_path)["path_rgb_0"]

    database_image_list = [f"{database_root / p}" for p in db_database_paths_raw]
    queries_image_list = [f"{query_root / p}" for p in db_queries_paths_raw]

    test_ds = TestDataset(
        database_image_list=database_image_list,
        queries_image_list=queries_image_list,
        rotation=rotation_angle,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )

    with torch.inference_mode():
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader, desc=f"Extracting descriptors for database images rotated by {rotation_angle} degrees"):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader, desc=f"Extracting descriptors for queries images rotated by {rotation_angle} degrees"):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

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
    print(f"Running VPR evaluation with method: {args.method}, rotation: {args.rot_360}, and yaml_data: {yaml_data}")

    D_0 = run_vpr(args, 0, yaml_data)
    if args.rot_360:
        D_90 = run_vpr(args, 90, yaml_data)
        D_180 = run_vpr(args, 180, yaml_data)
        D_270 = run_vpr(args, 270, yaml_data)
        D = np.minimum.reduce([D_0, D_90, D_180, D_270])
    else:
        D = D_0
    np.save(os.path.join(yaml_data["log_dir"], f"D.npy"), D)
