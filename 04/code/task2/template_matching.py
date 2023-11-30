import cv2
import numpy as np

def load_templates():
    emotions = ['happy', 'sad']
    templates = {'eye': {}, 'mouth': {}}
    for emotion in emotions:
        templates['eye'][emotion] = cv2.imread(f'Seb_{emotion}_eye2.png', 0)
        templates['mouth'][emotion] = cv2.imread(f'Seb_{emotion}_mouth2.png', 0)
    return templates

def extract_roi(image, keypoints, indices, padding=50):
    x_coords = [keypoints[i][0] for i in indices]
    y_coords = [keypoints[i][1] for i in indices]
    x_min, x_max = max(0, min(x_coords) - padding), min(image.shape[1], max(x_coords) + padding)
    y_min, y_max = max(0, min(y_coords) - padding), min(image.shape[0], max(y_coords) + padding)
    return image[y_min:y_max, x_min:x_max]

def match_template(roi, template):
    return cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

def determine_emotion(eye_roi, mouth_roi, templates):
    best_match = {'emotion': None, 'value': -1}
    for emotion in ['happy', 'sad']:
        eye_match = match_template(eye_roi, templates['eye'][emotion])
        mouth_match = match_template(mouth_roi, templates['mouth'][emotion])
        avg_match = np.mean([np.max(eye_match), np.max(mouth_match)])
        if avg_match > best_match['value']:
            best_match = {'emotion': emotion, 'value': avg_match}
    return best_match['emotion']

# Main execution block1
templates = load_templates()

# Define keypoints for the eyes and mouth regions (replace with actual keypoints)
keypoints = shape

# Extract ROIs and determine emotion
eye_roi = extract_roi(image, keypoints, range(36, 48))
mouth_roi = extract_roi(image, keypoints, range(48, 68))
emotion = determine_emotion(eye_roi, mouth_roi, templates)
print(f"The determined emotion is: {emotion}")
