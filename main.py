import cv2
import numpy as np
from pathlib import Path

def get_centroid(x, y, w, h):
    cx = x + int(w / 2)
    cy = y + int(h / 2)
    return cx, cy

count_line_y = 550  # Adjust based on your camera angle
offset = 6  # Pixel margin for the line
counter = 0

video_path = Path('media/vehant_hackathon_video_3.avi')
cap = cv2.VideoCapture(str(video_path))
algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=True)

# 2. Track seen IDs to prevent double counting (Simplified version)
temp_counter_list = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Process Mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 5)
    img_sub = algo.apply(blur)
    
    # Remove shadows (MOG2 marks shadows as 127 gray)
    _, th = cv2.threshold(img_sub, 254, 255, cv2.THRESH_BINARY)
    
    dilat = cv2.dilate(th, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(dilatada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.line(frame, (25, count_line_y), (1200, count_line_y), (255,127,0), 3)

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        
        # Lower these values if cars are far away/small
        if (w >= 50) and (h >= 50): 
            centroid = get_centroid(x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, centroid, 4, (0, 0, 255), -1)

            # Wider offset to catch fast moving cars
            if (centroid[1] < (count_line_y + offset)) and (centroid[1] > (count_line_y - offset)):
                counter += 1
                # Change line color briefly when triggered
                cv2.line(frame, (25, count_line_y), (1200, count_line_y), (0,0,255), 5)
                print(f"Detected! Total: {counter}")

    cv2.imshow("Detection Mask", dilatada) # CRITICAL FOR DEBUGGING
    cv2.imshow("Original Video", frame)
    
    if cv2.waitKey(1) == 27: break


cap.release()
cv2.destroyAllWindows()