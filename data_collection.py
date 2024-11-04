import cv2
import mediapipe as mp
import csv
import time

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize variables
data = []
paused = False
csv_file = "hand_signs_data.csv"


# Function to ask for a new label
def get_label():
    label = input("Enter the label name for this sign (or type 'exit' to quit): ")
    if label.lower() == 'exit':
        return None
    return label


# Create or open the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])

    # Start video capture
    cap = cv2.VideoCapture(0)
    label_name = get_label()

    while label_name:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Flip and process the frame
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)

            # Draw hand landmarks
            if result.multi_hand_landmarks and not paused:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract landmarks
                    landmarks = hand_landmarks.landmark
                    row = [label_name]
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z])

                    data.append(row)

            # Show the frame
            cv2.imshow("Hand Sign Data Collection", frame)

            # Key press handling
            key = cv2.waitKey(10)
            if key == ord(' '):  # Toggle pause
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('q'):  # Save and prompt for new label
                writer.writerows(data)
                print(f"Data for label '{label_name}' saved.")
                data.clear()
                label_name = get_label()
                if label_name is None:
                    break

        if label_name is None:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Data collection complete. Exiting.")

# Clean up Mediapipe resources
hands.close()
