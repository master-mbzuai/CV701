import numpy as np
import cv2
import time
import tensorflow as tf
import colorsys

SCORE_THRESHOLD = 0.1 # 0.1 if fp32 model, 10 if int8 is used
IMG_SZ=(224,224)

# Load the TFLite model.
#model_path = './04/code/task2/model.int8.tflite'
model_path = 'model.int8.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details, output_details)

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

prev_frame_time = 0

def model_inference(input=None):
    # Set input tensor.
    interpreter.set_tensor(input_details[0]['index'], input)

    # Run inference.
    interpreter.invoke()

    # Get output tensor.
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

def preprocess_img(frame):
    # img = frame[:, :, :]
    # maybe I can do    
    img = np.asarray(frame, dtype=np.float32)
    img = np.expand_dims(img,0)    
    return img

if __name__== "__main__":
    cap = cv2.VideoCapture(0)

    # Initialize the time and frames counter
    start_time = time.time()
    frames_counter = 0

    input_type = interpreter.get_input_details()[0]['dtype']
    print(input_type)

    with open("fps_log_optimized.txt", "w") as log_file:    

        while True:

            ret, frame = cap.read(0)
            if not ret:
                break

            # Crop the image
            cropped_image = frame[y_start:y_start+crop_size, x_start:x_start+crop_size]

            resized = cv2.resize(cropped_image, IMG_SZ)

            #frame = cv2.resize(cap.read()[1], IMG_SZ, interpolation=cv2.INTER_LINEAR)
            input_img = preprocess_img(resized)
            
            # input_img = np.transpose(input_img, (0, 3, 1, 2))
            # print(input_img.shape)

            output = model_inference(input_img)

            keypoints = output.reshape(-1, 2) 

            # print(output)
            # print(keypoints)
            # print(output[0][4])
            # frame = post_process(frame, output[0], score_threshold=SCORE_THRESHOLD)
            # Increment the frames counter for each frame read
            frames_counter += 1

            # Calculate the actual FPS

            # Time when we finish processing for this frame
            new_frame_time = time.time()
            inference_time = new_frame_time-prev_frame_time
            prev_frame_time = new_frame_time
                    
            elapsed_time = time.time() - start_time
            calculated_fps = frames_counter / elapsed_time

            log_file.write(f"{inference_time} - {calculated_fps}\n")

            for i in range(68):
                cv2.circle(resized, (int(keypoints[i][0]), int(keypoints[i][1])), 2, (0, 255, 0), 3)

            # Display the FPS on the frame
            cv2.putText(resized, f"FPS : {calculated_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(resized, f"IT : {inference_time:.2f}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow('out',resized)
            cv2.waitKey(1)