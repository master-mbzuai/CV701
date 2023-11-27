import cv2
import torch
import time
from models.model1_pre import FacialPoints

from micromind.utils.parse import parse_arguments

def main():
    # Capture video from the first camera source
    cap = cv2.VideoCapture(0)

    hparams = parse_arguments()    
    d = hparams.d

    m = FacialPoints(hparams)    

    while True:

        ## count fps
        frame_count = 0
        start_time = time.time()    

        # Read a new frame
        ret, frame = cap.read()

        if not ret:
            break

        # crop the frame to 1080 x 1080
        crop_frame = frame[0:1080, 640:1720]        

        # Resize the frame to 224x224 pixels
        resized_frame = cv2.resize(frame, (224, 224))


        m.modules.eval()
        res = m.forward([torch.Tensor(resized_frame).permute(2,1,0).view(1,3,224,224)])

        res = res * 1080/224
        res = res.view(68, 2)

        for i in range(68):
            cv2.circle(crop_frame, (int(res[i][0]), int(res[i][1])), 1, (0, 0, 255), 10)       

        frame_count += 1                                         

        ## count fps        
        # Calculate FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Display FPS on the frame
        cv2.putText(crop_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display the Inference time on the frame
        cv2.putText(crop_frame, f"Inference time: {round(elapsed_time,2)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Resized Frame', crop_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
