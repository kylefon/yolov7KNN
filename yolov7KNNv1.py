import cv2
import torch
import joblib
import pytesseract
from picamera2 import Picamera2
import numpy as np
import time
import tkinter as tk
from PIL import Image, ImageTk
import os

# Load YOLOv7 model
model = torch.hub.load('/home/fonsecamartinez/Documents/yolov7', 'custom', './last-v7-oct15.pt')

# Load the saved KNN model
knn_model = joblib.load('KNN_11-30-2024v1.sav')

# Function to extract color histogram features
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Function to apply edge detection (Canny)
def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function to apply Local Binary Pattern (LBP) features
def extract_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = cv2.localBinaryPattern(gray, 8, 1, cv2.LBP_VARIANT_UNIFORM)
    lbp_hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
    cv2.normalize(lbp_hist, lbp_hist)
    return lbp_hist.flatten()

# Function to perform OCR and return text
def perform_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    license_plate_text = pytesseract.image_to_string(processed_image, config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    return license_plate_text.strip()

# Simplify and map labels based on the classifier output
def simplify_label(label):
    if 'private' in label:
        return 'private'
    elif 'public' in label:
        return 'public'
    elif 'government' in label:
        return 'government'
    return 'unknown'

# Apply heuristic rules to refine classifications
def apply_heuristic_rules(roi, simplified_label, license_plate_text, confidence):
    # 1. Government Plates - Prioritize red-based detections
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_hue_ratio = np.sum((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 10)) / (hsv.size / 3)  # Check red hue range
    if red_hue_ratio > 0.4:  # Threshold to indicate red dominance
        simplified_label = 'government'

    # 2. Special Plates - Use OCR confidence and ensure the detected number is between 1 and 17
    if simplified_label == 'private' and license_plate_text.isdigit():
        plate_number = int(license_plate_text)
        if 1 <= plate_number <= 17 and confidence > 80:  # Example: confidence threshold of 80%
            simplified_label = 'special'

    # 3. Private/Public Plates - Check yellow dominance and prefer public if ambiguous
    if simplified_label == 'private':
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        yellow_hue_ratio = np.sum((hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 40)) / (hsv.size / 3)  # Check yellow hue range
        if yellow_hue_ratio > 0.4:  # Threshold to indicate yellow dominance
            simplified_label = 'public'

    return simplified_label

# Web cam function to capture and process frames
def web_cam_func(max_fps=1.0):
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()

    min_frame_duration = 1.0 / max_fps

    # Define output folders
    output_folders = {
        'private': 'detections/private',
        'public': 'detections/public',
        'government': 'detections/government',
        'special': 'detections/special'
    }
    
    # Create folders if they don't exist
    for folder in output_folders.values():
        os.makedirs(folder, exist_ok=True)

    while not quit_program:
        start_time = time.time()
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        results = model(frame)

        detection_made = False  # Track if a detection is made
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.6:
                detected_label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)

                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                roi = frame[y1:y2, x1:x2]

                # Extract features: Color histogram, edges, and LBP
                roi_features = np.concatenate([
                    extract_color_histogram(roi), 
                    extract_edges(roi).flatten(), 
                    extract_lbp(roi)
                ]).reshape(1, -1)

                predicted_label = knn_model.predict(roi_features)
                simplified_label = simplify_label(predicted_label[0])

                if simplified_label == 'private':
                    license_plate_text = perform_ocr(roi)
                    # Apply heuristic rules for further refinement
                    simplified_label = apply_heuristic_rules(roi, simplified_label, license_plate_text, conf)

                cv2.putText(frame, f'Label: {simplified_label}', (int(xyxy[0]), int(xyxy[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if simplified_label == 'private':
                    cv2.putText(frame, f'OCR: {license_plate_text}', (int(xyxy[0]), int(xyxy[1])-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Set detection flag to True
                detection_made = True

                # Save the frame to the respective folder
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = os.path.join(output_folders[simplified_label], f"detected_frame_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

        cv2.putText(frame, f'FPS: {max_fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        yield frame

        elapsed_time = time.time() - start_time
        if elapsed_time < min_frame_duration:
            time.sleep(min_frame_duration - elapsed_time)

    picam2.stop()

# Update the frame in the GUI
def update_frame():
    try:
        frame = next(video_feed)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)

        panel.imgtk = imgtk
        panel.config(image=imgtk)

    except StopIteration:
        return

    if not quit_program:
        panel.after(30, update_frame)

# Start inference button callback
def start_inference():
    global video_feed, quit_program
    quit_program = False

    start_button.pack_forget()
    quit_button.pack_forget()

    global panel
    panel = tk.Label(root)
    panel.pack()

    quit_button.pack()
    video_feed = web_cam_func()
    update_frame()

# Quit program callback
def quit_program_func():
    global quit_program
    quit_program = True
    root.quit()

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("License Plate Detection")
root.attributes("-fullscreen", True)
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

start_button = tk.Button(root, text="Start", command=start_inference, padx=20, pady=10)
start_button.pack()

quit_button = tk.Button(root, text="Quit", command=quit_program_func, padx=20, pady=10)
quit_button.pack()

quit_program = False
root.mainloop()