import cv2
import time
import torch
import numpy as np

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change '0' to '-1' if '0' does not work

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

prev_frame_time = 0
new_frame_time = 0

prev_frame_time = 0

# Original image dimensions
original_height, original_width = 1080, 1920

# Size of the square crop
crop_size = 1080

# Ensure the original image is large enough
if original_height < crop_size or original_width < crop_size:
    raise ValueError("The original image is smaller than the cropping size.")

# Calculate the x and y coordinates for the crop
x_start = (original_width - crop_size) // 2
y_start = (original_height - crop_size) // 2

hparams = parse_arguments()             

m = FacialPoints(hparams)
m.modules.eval()

with open("fps_log.txt", "w") as log_file:    
    with torch.no_grad():
        while True:
            ret, frame = cap.read(1)
            if not ret:
                break
            
            # Crop the image
            cropped_image = frame[y_start:y_start+crop_size, x_start:x_start+crop_size]

            # Convert the entire original image to grayscale
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image back to BGR (to match color space)
            gray_image_colored = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Replace the cropped area in the grayscale image with the cropped color image
            gray_image_colored[y_start:y_start+crop_size, x_start:x_start+crop_size] = cropped_image


            ## MODEL INFERENCE
            
            cropped_image = cv2.resize(cropped_image, (224, 224))
            cropped_image = np.array(cropped_image / 255.0).astype(np.float32)

            input = [torch.tensor(cropped_image).permute(2, 0, 1).view(1, 3, 224, 224)]            

            m.forward(input)

            ## SHOWING THINGS ON SCREEN

            # Time when we finish processing for this frame
            new_frame_time = time.time()

            # Set camera properties for 60 FPS
            cap.set(cv2.CAP_PROP_FPS, 60)

            # Calculating the fps
            inference_time = str(new_frame_time-prev_frame_time)

            # keep only 2 decimal digits
            inference_time = inference_time[:inference_time.index('.')+3]

            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            # Convert the fps to string so that we can display it on frame
            fps = str(int(fps))

            # Put fps on the frame
            cv2.putText(gray_image_colored, inference_time, (7, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(gray_image_colored, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Frame', gray_image_colored)

            log_file.write(f"{new_frame_time} - {fps}\n")

            # Break the loop with the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
