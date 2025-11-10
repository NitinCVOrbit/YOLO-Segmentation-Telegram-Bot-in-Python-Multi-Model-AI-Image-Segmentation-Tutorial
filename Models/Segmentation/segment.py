from ultralytics import YOLO
import cv2
import os
import numpy as np


def draw_segmentation(image, mask, class_colors):

    # Define transparency level (0 to 1)
    alpha = 0.5  # 50% transparency

    # Create a colored overlay of the same shape as the image
    overlay = np.zeros_like(image, dtype=np.uint8)

    for class_id in range(len(class_colors)):
        if class_id == 0:
            continue  # Skip background class
        class_mask = (mask == class_id)
        color = class_colors[class_id]
        overlay[class_mask] = color

    # ✅ Fixed: Convert PIL image to NumPy before resizing
    image = np.array(image)  # Convert PIL to NumPy

    # Create a copy to preserve original image
    blended_image = image.copy()
    
    # Create a mask where any class ≠ 0
    object_mask = (mask != 0)   

    # Blend for all class masks
    blended_image[object_mask] = cv2.addWeighted(
        image[object_mask], 1 - alpha, overlay[object_mask], alpha, gamma = 0
    )    

    return blended_image


def segmentation(image, class_colors, weights):
    
    # Load model weights
    weights_path = os.path.join(os.path.dirname(__file__),"..","Weights",f"{weights}")

    model = YOLO(weights_path)
    
    # Perform inference
    results = model(image)[0]

    image = np.array(image)
    # print(results)

    if results.masks:
      # Draw segmentation masks
      for seg in results.masks.data:
          mask = seg.cpu().numpy()
          # Resize mask to match image dimensions
          mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
          image = draw_segmentation(image, mask_resized, class_colors)       

    return image

