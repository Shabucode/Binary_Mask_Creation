#pip install opencv-python numpy
#!pip install ultralytics -q

#opencv-python 
#numpy
#ultralytics in requirements.txt

from ultralytics import YOLO
import cv2
import numpy as np
#from google.colab.patches import cv2_imshow
import os
import matplotlib.pyplot as plt

# def generate_masked_image(image_path, model_path="yolov8m-seg.pt"):
#     # Load model
#     model = YOLO(model_path)
    
#     # Perform prediction
#     predict = model.predict(image_path, save=True, save_txt=True)
    
#     # Read the original image to get its shape
#     H, W, _ = cv2.imread(image_path).shape
    
#     # Extract the first mask from predictions
#     m = (predict[0].masks.data[0].numpy() * 255).astype("uint8")  # Use data[0] if only one mask is present
    
#     # Create 'masks' directory if it does not exist
#     os.makedirs('masks', exist_ok=True)
    
#     # Save the mask image
#     mask_filename = os.path.splitext(os.path.basename(image_path))[0] + '_mask.jpg'
#     mask_path = os.path.join('masks', mask_filename)
#     cv2.imwrite(mask_path, m)
    
#     print(f"Mask saved at: {mask_path}")
    
#     # Optionally, display the mask using matplotlib (commented out)
#     # plt.imshow(m, cmap='gray')
#     # plt.show()
    
#     return mask_path

def generate_combined_masked_image(image_path, model_path="yolov8m-seg.pt"):
    # Load model
    model = YOLO(model_path)
    
    # Perform prediction
    predict = model.predict(image_path, save=True, save_txt=True)
    
    # Check if any masks are present
    if predict[0].masks is None:
        raise ValueError("No masks found in the prediction.")
    
    # Initialize the combined mask with the first mask
    combined_mask = predict[0].masks.data[0].numpy()
    
    # Iterate over all masks and combine them
    for i in range(1, predict[0].masks.data.shape[0]):
        mask = predict[0].masks.data[i].numpy()
        combined_mask = np.maximum(combined_mask, mask)
    
    # Scale combined mask to uint8 for visualization
    combined_mask_scaled = (combined_mask * 255).astype("uint8")
    
    # Create 'masks' directory if it does not exist
    os.makedirs('masks', exist_ok=True)
    
    # Save the combined mask image
    mask_filename = os.path.splitext(os.path.basename(image_path))[0] + '_combined_mask.png'
    mask_path = os.path.join('masks', mask_filename)
    cv2.imwrite(mask_path, combined_mask_scaled)
    
    print(f"Combined mask saved at: {mask_path}")
    
    # Optionally, display the combined mask using matplotlib (commented out)
    # plt.imshow(combined_mask_scaled, cmap='gray')
    # plt.show()
    
    return combined_mask_scaled

# Example usage
image_path = input("enter image path: ")
mask_path = generate_combined_masked_image(image_path)