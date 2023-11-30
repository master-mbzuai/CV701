import cv2
import time
import torch
import numpy as np

from models.model1_pre import FacialPoints
from micromind.utils.parse import parse_arguments

from ptflops import get_model_complexity_info

## Emotions detection

import cv2
import numpy as np

def extract_region_of_interest(keypoints, image, region_indices, padding=50):

    #print(keypoints)
    # Compute the bounding box for the region of interest
    x_coordinates = [point[0] for point in keypoints[region_indices]]
    y_coordinates = [point[1] for point in keypoints[region_indices]]

    print(x_coordinates, y_coordinates)

    #print(image.shape[0])
    x_min, x_max = max(0, min(x_coordinates) - padding), min(image.shape[1], max(x_coordinates) + padding)
    y_min, y_max = max(0, min(y_coordinates) - padding), min(image.shape[0], max(y_coordinates) + padding)

    # round to integer

    x_min = int(x_min)
    x_max = int(x_max)
    y_min = int(y_min)
    y_max = int(y_max)
    
    print(y_max, y_min, x_max, x_min)

    # Crop the region of interest from the image
    return image[y_min:y_max, x_min:x_max], (x_max, y_max), (x_min, y_min) 

def match_templates(roi, templates, method=cv2.TM_CCOEFF_NORMED):
    print("roi", roi.shape)
    max_vals = []
    roi = roi.astype(np.float32)  # Convert ROI to float32 if it's not already
    if(roi.shape[0] != 0 and roi.shape[1] != 0):
        for template in templates:
            templ_resized = cv2.resize(template, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)
            print("temp", templ_resized.shape)
            templ_resized = templ_resized.astype(np.float32)  # Ensure template is float32
            if len(roi.shape) > 2 and roi.shape[2] == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale if it's colored
            if len(templ_resized.shape) > 2 and templ_resized.shape[2] == 3:
                templ_resized = cv2.cvtColor(templ_resized, cv2.COLOR_BGR2GRAY)  # Convert template to grayscale if it's colored
            res = cv2.matchTemplate(roi, templ_resized, method)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            max_vals.append(max_val)
    return max_vals

def determine_emotion(eye_roi, mouth_roi, templates):
    # Compare the ROIs with the templates
    eye_matches = match_templates(eye_roi, templates['eye'])
    mouth_matches = match_templates(mouth_roi, templates['mouth'])
    
    # Average match values for eyes and mouth
    average_matches = np.mean([eye_matches, mouth_matches], axis=0)

    # Determine emotion with the highest average match value
    emotions = ['happy', 'sad']
    best_match_index = np.argmax(average_matches)
    
    return emotions[best_match_index]

# Load templates for different emotions (for both eyes and mouth)
templates = {
    'eye': {
        'happy': cv2.imread('./data/Seb_Happy_eye2.png', 0),
        'sad': cv2.imread('./data/Seb_Sad_eye2.png', 0),

    },
    'mouth': {
        'happy': cv2.imread('./data/Seb_Happy_mouth2.png', 0),
        'sad': cv2.imread('./data/Seb_Sad_mouth2.png', 0),
        
    }
}

# Flatten the templates dictionary to a list while preserving order
eye_templates = [templates['eye'][emotion] for emotion in ['happy', 'sad']]
mouth_templates = [templates['mouth'][emotion] for emotion in ['happy', 'sad']]

# Extract ROIs for eyes and mouth from the image
eye_indices = list(range(36, 48))  # Indices for eye keypoints
mouth_indices = list(range(48, 68))  # Indices for mouth keypoints






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
            resized = np.asarray(resized, dtype=np.float32)
            resized = np.expand_dims(resized,0)

            # input = [torch.tensor(resized).permute(2, 1, 0).view(1, 3, 224, 224)]            
            input = [torch.tensor(resized).permute(0, 3, 2, 1)]            

            keypoints = m.forward(input)

            keypoints = keypoints.view(-1, 2)
            keypoints = keypoints * 1080/224

            #print(keypoints)

            ## EMOTIONS DETECTION

            # Load your image and convert it to grayscale if necessary
            image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY) if len(cropped_image.shape) == 3 else cropped_image

            eye_roi, max_roi_eye, min_roi_eye = extract_region_of_interest(np.array(keypoints, dtype=np.float32), image, eye_indices)
            mouth_roi, max_roi, min_roi = extract_region_of_interest(np.array(keypoints, dtype=np.float32), image, mouth_indices)

            cv2.rectangle(cropped_image, max_roi, min_roi, (0, 255, 0), 2)
            cv2.rectangle(cropped_image, max_roi_eye, min_roi_eye, (0, 255, 0), 2)

            # Determine emotion by comparing the extracted ROIs with the templates
            emotion = determine_emotion(eye_roi, mouth_roi, {'eye': eye_templates, 'mouth': mouth_templates})
            cv2.putText(cropped_image, emotion, (7, 210), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

            ## SHOWING THINGS ON SCREEN            

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
