import cv2
import time
import torch
import numpy as np

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments

from ptflops import get_model_complexity_info

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


# backbone
# print(m.modules["feature_extractor"].get_MAC())
# print(m.modules["feature_extractor"].get_params())

# classifier

# network information
# flop, param = get_model_complexity_info(m.modules["classifier"], (m.input, 1, 1), as_strings=False,
#                                            print_per_layer_stat=False, verbose=False)

# tot_mac = m.modules["feature_extractor"].get_MAC() + flop/2
# tot_param = m.modules["feature_extractor"].get_params() + param

# print(tot_mac, tot_param)


with open("fps_log.txt", "w") as log_file:    
    with torch.no_grad():
        while True:
            ret, frame = cap.read(0)
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

            resized = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_LINEAR)
            # resized = np.array(resized).astype(np.float32)
            resized = np.asarray(resized, dtype=np.float16)
            resized = np.expand_dims(resized,0)

            # input = [torch.tensor(resized).permute(2, 1, 0).view(1, 3, 224, 224)]            
            input = [torch.tensor(resized).permute(0, 3, 2, 1)]            

            keypoints = m.forward(input)

            ## SHOWING THINGS ON SCREEN            

            keypoints = keypoints.view(-1, 2)
            keypoints = keypoints * 1080/224

            for i in range(68):
                cv2.circle(cropped_image, (int(keypoints[i][0]), int(keypoints[i][1])), 2, (0, 255, 0), 3)

            # Time when we finish processing for this frame
            new_frame_time = time.time()

            # Calculating the fps
            inference_time = str(new_frame_time-prev_frame_time)

            # keep only 2 decimal digits
            inference_time = inference_time[:inference_time.index('.')+3]

            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time

            # Convert the fps to string so that we can display it on frame
            fps = str(int(fps))

            # Put fps on the frame
            cv2.putText(cropped_image, inference_time, (7, 140), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.putText(cropped_image, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Frame', cropped_image)

            log_file.write(f"{new_frame_time} - {fps}\n")

            # Break the loop with the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Release and destroy all windows before termination
cap.release()
cv2.destroyAllWindows()
