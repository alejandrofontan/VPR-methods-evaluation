import os
import cv2
from tqdm import tqdm

def rotate_and_save_images(database_image_list, database_image_list_rotated, angle = 0):
    for filename in tqdm(zip(database_image_list, database_image_list_rotated), desc=f"Rotating images by {angle} degrees"):
        # Load image
        img = cv2.imread(filename[0])
        if img is None:
            print(f"Skipping {filename[0]}, could not load.")
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

        # Save image
        cv2.imwrite(filename[1], rotated)