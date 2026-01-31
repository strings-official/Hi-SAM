import os
import json
import cv2
import numpy as np
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def process_single_image(image_name, image_folder, jsonl_folder, mask_folder, vis_folder, dilation_kernel_size):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        return

    image_path = os.path.join(image_folder, image_name)
    jsonl_path = os.path.join(jsonl_folder, os.path.splitext(image_name)[0] + '.jsonl')
    image_id = os.path.splitext(image_name)[0]

    if not os.path.exists(jsonl_path):
        print(f"No JSONL file found for {image_name}")
        return

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Failed to read image {image_name}")
        return

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            if data['image_id'] == image_id:
                for word in data['words']:
                    vertices = word.get('vertices')
                    if vertices:
                        pts = np.array(vertices, dtype=np.int32)
                        cv2.fillPoly(mask, [pts], color=255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    mask_path = os.path.join(mask_folder, image_name)
    cv2.imwrite(mask_path, dilated_mask)
    print(f"Mask saved to {mask_path}")

    vis_image = image.copy()
    orange_overlay = np.full_like(vis_image, (0, 165, 255))  # Orange BGR
    mask_colored = cv2.bitwise_and(orange_overlay, orange_overlay, mask=dilated_mask)
    vis_image = cv2.addWeighted(vis_image, 1.0, mask_colored, 0.5, 0)

    vis_path = os.path.join(vis_folder, image_name)
    cv2.imwrite(vis_path, vis_image)
    print(f"Visualization saved to {vis_path}")

def process_image(image_folder, jsonl_folder, output_folder, dilation_kernel_size=5, num_workers=8):
    os.makedirs(output_folder, exist_ok=True)
    mask_folder = os.path.join(output_folder, 'masks')
    vis_folder = os.path.join(output_folder, 'masks_vis')
    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(vis_folder, exist_ok=True)

    image_names = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        executor.map(partial(
            process_single_image,
            image_folder=image_folder,
            jsonl_folder=jsonl_folder,
            mask_folder=mask_folder,
            vis_folder=vis_folder,
            dilation_kernel_size=dilation_kernel_size
        ), image_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate binary masks and visualizations from JSONL annotations.")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--dilation', type=int, default=225, help="Kernel size for dilation.")
    
    args = parser.parse_args()
    
    source_path = args.dataset_path

    process_image(
        source_path + '/images',
        source_path + '/hisam_jsons',
        source_path,
        dilation_kernel_size=args.dilation,
    )
    