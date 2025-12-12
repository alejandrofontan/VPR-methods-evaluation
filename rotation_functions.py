import os
import cv2
from tqdm import tqdm

def rotate_and_save_images(input_folder, output_folder, angle = 0):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all files in the input folder
    for filename in tqdm(os.listdir(input_folder)):
        file_path = os.path.join(input_folder, filename)

        # Only process image files
        if not (filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))):
            continue

        # Load image
        img = cv2.imread(file_path)
        if img is None:
            print(f"Skipping {filename}, could not load.")
            continue

        # Save rotated versions
        if angle == 0:
            rotated = img
        if angle == 90:
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
             rotated = cv2.rotate(img, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Create output filename
        name, ext = os.path.splitext(filename)
        out_path = os.path.join(output_folder, filename)

        # Save image
        cv2.imwrite(out_path, rotated)
