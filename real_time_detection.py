import cv2
import mediapipe as mp
import numpy as np
import pickle
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk

# Load the pre-trained model and label encoder
with open('best_rf_sign_language_model.pkl', 'rb') as model_file:
    sign_language_model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start video capture from the webcam
video_capture = cv2.VideoCapture(0)

def predict_sign(landmarks):
    """Predict the sign from hand landmarks using the trained model."""
    reshaped_landmarks = np.array(landmarks).reshape(1, -1)  # Reshape for model input
    prediction = sign_language_model.predict(reshaped_landmarks)
    sign_label = label_encoder.inverse_transform(prediction)[0]
    return sign_label

# Create the main window
window = tk.Tk()
window.title("Sign Language Detection")
window.geometry("800x600")
window.configure(bg="black")

# Create a frame for the UI elements
frame = Frame(window, bg="black", bd=10, relief="ridge")
frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

# Create a label to display the predicted sign
sign_label = Label(frame, text="Sign: ", font=("Helvetica", 30, "bold"), bg="lightblue", fg="darkblue")
sign_label.pack(pady=20)

# Create a canvas to display the video feed
canvas = tk.Canvas(frame, width=640, height=480)
canvas.pack()

# Variable to control the detection state
is_detecting = False

def start_detection():
    """Start the detection process."""
    global is_detecting
    is_detecting = True
    update_frame()  # Start updating frames

def update_frame():
    """Capture the video frame, process it, and update the UI."""
    global is_detecting
    if is_detecting:
        success, frame = video_capture.read()
        if success:
            # Convert the frame to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)

            # If hands are detected, process each hand's landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        # Append x, y, z coordinates of each landmark
                        landmarks.extend([landmark.x, landmark.y, landmark.z])

                    # Predict the sign and display it
                    predicted_sign = predict_sign(landmarks)
                    sign_label.config(text=f"Sign: {predicted_sign}")  # Update the label text
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert frame to ImageTk format and display it on the canvas
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, image=img_tk, anchor=tk.NW)
            canvas.image = img_tk  # Keep a reference to avoid garbage collection

        # Schedule the next frame update
        window.after(10, update_frame)

# Create a button to start detection
start_button = tk.Button(frame, text="Start Detection", command=start_detection, font=("Helvetica", 16), bg="green", fg="white")
start_button.pack(pady=10)

# Create an exit button
def exit_app():
    """Exit the application gracefully."""
    global is_detecting
    is_detecting = False
    video_capture.release()
    window.destroy()

exit_button = tk.Button(frame, text="Exit", command=exit_app, font=("Helvetica", 16), bg="red", fg="white")
exit_button.pack(pady=10)

# Start the GUI main loop
window.mainloop()
