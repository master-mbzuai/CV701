import cv2
import numpy as np

def extract_region_of_interest(keypoints, image, region_indices, padding=50):
    # Compute the bounding box for the region of interest
    x_coordinates = [point[0] for point in keypoints[region_indices]]
    y_coordinates = [point[1] for point in keypoints[region_indices]]
    x_min, x_max = max(0, min(x_coordinates) - padding), min(image.shape[1], max(x_coordinates) + padding)
    y_min, y_max = max(0, min(y_coordinates) - padding), min(image.shape[0], max(y_coordinates) + padding)
    
    # Crop the region of interest from the image
    return image[y_min:y_max, x_min:x_max]

def match_templates(roi, templates, method=cv2.TM_CCOEFF_NORMED):
    max_vals = []
    roi = roi.astype(np.float32)  # Convert ROI to float32 if it's not already
    for template in templates:
        template = template.astype(np.float32)  # Ensure template is float32
        if len(roi.shape) > 2 and roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert ROI to grayscale if it's colored
        if len(template.shape) > 2 and template.shape[2] == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)  # Convert template to grayscale if it's colored
        res = cv2.matchTemplate(roi, template, method)
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

# Load your image and convert it to grayscale if necessary
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

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

# Define the keypoints for the eyes and mouth regions
# Replace these with the actual keypoints
keypoints = shape

# Extract ROIs for eyes and mouth from the image
eye_indices = list(range(36, 48))  # Indices for eye keypoints
mouth_indices = list(range(48, 68))  # Indices for mouth keypoints
eye_roi = extract_region_of_interest(keypoints, image, eye_indices)
mouth_roi = extract_region_of_interest(keypoints, image, mouth_indices)

# Determine emotion by comparing the extracted ROIs with the templates
emotion = determine_emotion(eye_roi, mouth_roi, {'eye': eye_templates, 'mouth': mouth_templates})
print(f"The determined emotion is: {emotion}")
