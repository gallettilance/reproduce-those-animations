import sys
import cv2
import argparse
import numpy as np

def change_color(video_path, output_path, target_color, replacement_color):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the codec information and frame size
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range of target color in HSV
        lower_color = np.array(target_color[0], dtype=np.uint8)
        upper_color = np.array(target_color[1], dtype=np.uint8)
        
        # Threshold the HSV image to get only target colors
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Replace target color with replacement color
        frame[mask > 0] = replacement_color
        
        # Write the frame into the output video
        out.write(frame)
        
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="description")
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--target", type=set, default=([0, 0, 200], [179, 50, 255]), required=False)
    parser.add_argument("--replacement", type=list, default=[102, 220, 255], required=False)
    args=parser.parse_args()
    change_color(args.input, args.output, args.target, args.replacement)