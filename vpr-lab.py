
import parser
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import vpr_models
from rotation_functions import rotate_and_save_images
from test_dataset import TestDataset

def run_vpr(args, rotation_angle):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    #logger.remove()  # Remove possibly previously existing loggers
    #log_dir = Path("logs") / args.log_dir # / start_time.strftime("%Y-%m-%d_%H-%M-%S")
    #logger.add(sys.stdout, colorize=True, format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    #logger.add(log_dir / "info.log", format="<green>{time:%Y-%m-%d %H:%M:%S}</green> {message}", level="INFO")
    #logger.add(log_dir / "debug.log", level="DEBUG")
    #logger.info(" ".join(sys.argv))
    #logger.info(f"Arguments: {args}")
    #logger.info(
    #    f"Testing with {args.method} with a {args.backbone} backbone and descriptors dimension {args.descriptors_dimension}"
    #)
    #logger.info(f"The outputs are being saved in {log_dir}")

    model = vpr_models.get_model(args.method, args.backbone, args.descriptors_dimension)
    model = model.eval().to(args.device)

    input_folder = args.database_folder
    output_folder = input_folder + f"_{rotation_angle}"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        if rotation_angle == 0:
            for item_path in Path(input_folder).iterdir():
                link_path = Path(output_folder) / item_path.name
                link_path.symlink_to(item_path)
        else:
            rotate_and_save_images(input_folder, output_folder, rotation_angle)

    test_ds = TestDataset(
        output_folder,
        args.queries_folder,
        positive_dist_threshold=args.positive_dist_threshold,
        image_size=args.image_size,
        use_labels=args.use_labels,
    )
    
    with torch.inference_mode():
        #logger.debug("Extracting database descriptors for evaluation/testing")
        database_subset_ds = Subset(test_ds, list(range(test_ds.num_database)))
        database_dataloader = DataLoader(
            dataset=database_subset_ds, num_workers=args.num_workers, batch_size=args.batch_size
        )
        all_descriptors = np.empty((len(test_ds), args.descriptors_dimension), dtype="float32")
        for images, indices in tqdm(database_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

        #logger.debug("Extracting queries descriptors for evaluation/testing using batch size 1")
        queries_subset_ds = Subset(
            test_ds, list(range(test_ds.num_database, test_ds.num_database + test_ds.num_queries))
        )
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers, batch_size=1)
        for images, indices in tqdm(queries_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            all_descriptors[indices.numpy(), :] = descriptors

    queries_descriptors = all_descriptors[test_ds.num_database :]
    database_descriptors = all_descriptors[: test_ds.num_database]

    # Compute similarity matrix
    #logger.debug("Calculating similarity matrix")

    # Prepare for manual ordering of images for similarity matrix
    query_path = Path(args.queries_folder).parent
    database_path = Path(input_folder).parent
    queries_rgb_csv_path = os.path.join(query_path, 'rgb.csv')
    database_rgb_csv_path = os.path.join(database_path, 'rgb.csv')

    query_df = pd.read_csv(queries_rgb_csv_path)
    db_df = pd.read_csv(database_rgb_csv_path)
    
    prefix = Path(input_folder).name
    db_df['path_rgb0'] = db_df['path_rgb0'].str.replace(prefix, f"rgb_0_{rotation_angle}", regex=False)
    
    q_indices = []
    for image in query_df['path_rgb0']:
        filename = os.path.join(query_path, image)
        if filename in test_ds.queries_paths:
            q_idx = test_ds.queries_paths.index(filename)
            q_indices.append(q_idx)

    db_indices = []
    for image in db_df['path_rgb0']:
        filename = os.path.join(database_path, image)
        if filename in test_ds.database_paths:
            db_idx = test_ds.database_paths.index(filename)
            db_indices.append(db_idx)

    distance_matrix = faiss.pairwise_distances(database_descriptors, queries_descriptors)
    
    distance_matrix_sorted = np.zeros_like(distance_matrix)
    for i, q_idx in enumerate(q_indices):
        for j, db_idx in enumerate(db_indices):
            distance_matrix_sorted[j,i] = distance_matrix[db_idx, q_idx]

    np.save(os.path.join(log_dir, f"D_{rotation_angle}_{args.method}.npy"), distance_matrix_sorted)  # saves binary numpy file

    if args.verbose:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Distance")
        plt.xlabel("Query Descriptors")
        plt.ylabel("Database Descriptors")
        plt.title("Pairwise Distance Matrix")

        plt.show()

if __name__ == "__main__":
    args = parser.parse_arguments()
    run_vpr(args, 0)
    run_vpr(args, 90)
    run_vpr(args, 180)
    run_vpr(args, 270)