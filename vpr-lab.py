
import parser
import os
import yaml
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

def run_vpr(args, rotation_angle, yaml_data):   
    input_folder = yaml_data['rgb_folder_db']
    queries_folder = yaml_data['rgb_folder_q']
    database_rgb_csv_path = yaml_data['rgb_list_db']
    queries_rgb_csv_path = yaml_data['rgb_list_q']
    query_path = Path(queries_folder).parent
    database_path = Path(input_folder).parent

    log_dir = Path(yaml_data['log_dir'])
    log_dir.mkdir(exist_ok=True, parents=True)
    
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

    #input_folder = args.database_folder
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
        queries_folder,
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

    query_df = pd.read_csv(queries_rgb_csv_path)
    db_df = pd.read_csv(database_rgb_csv_path)
    
    prefix = Path(input_folder).name
    db_df['path_rgb_0'] = db_df['path_rgb_0'].str.replace(prefix, f"rgb_0_{rotation_angle}", regex=False)
    
    q_indices = []
    for image in query_df['path_rgb_0']:
        filename = os.path.join(query_path, image)
        if filename in test_ds.queries_paths:
            q_idx = test_ds.queries_paths.index(filename)
            q_indices.append(q_idx)

    db_indices = []
    for image in db_df['path_rgb_0']:
        filename = os.path.join(database_path, image)
        if filename in test_ds.database_paths:
            db_idx = test_ds.database_paths.index(filename)
            db_indices.append(db_idx)

    distance_matrix = faiss.pairwise_distances(database_descriptors, queries_descriptors)
    
    distance_matrix_sorted = np.zeros_like(distance_matrix)
    for i, q_idx in enumerate(q_indices):
        for j, db_idx in enumerate(db_indices):
            distance_matrix_sorted[j,i] = distance_matrix[db_idx, q_idx]

    np.save(os.path.join(log_dir, f"D_{rotation_angle}.npy"), distance_matrix_sorted)  # saves binary numpy file

    if args.verbose:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.imshow(distance_matrix, cmap="viridis", aspect="auto")
        plt.colorbar(label="Distance")
        plt.xlabel("Query Descriptors")
        plt.ylabel("Database Descriptors")
        plt.title("Pairwise Distance Matrix")

        plt.show()
    return distance_matrix_sorted

if __name__ == "__main__":
    args = parser.parse_arguments()

    exp_yaml = args.exp_yaml
    with open(exp_yaml, 'r') as stream:
        yaml_data = yaml.safe_load(stream)

    D_0 = run_vpr(args, 0, yaml_data)
    D_90 =  run_vpr(args, 90, yaml_data)
    D_180 = run_vpr(args, 180, yaml_data)
    D_270 = run_vpr(args, 270, yaml_data)
    D = np.minimum.reduce([D_0, D_90, D_180, D_270])
    np.save(os.path.join(yaml_data['log_dir'], f"D.npy"), D)
