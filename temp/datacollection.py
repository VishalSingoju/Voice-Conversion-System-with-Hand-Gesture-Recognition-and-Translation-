import mediapipe as mp
import cv2
import csv

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open a CSV file to save gesture data
csv_file = "custom_gesture_dataset.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header row
    header = ["label"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)]
    writer.writerow(header)

    # Start webcam capture
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit and 's' to save gesture data.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture image.")
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw landmarks on the video frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])  # Add normalized x, y coordinates

                # Display and allow the user to label the gesture
                cv2.putText(frame, "Press 's' to save this gesture.", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Save to CSV when 's' is pressed
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    label = input("Enter the label for this gesture: ")
                    writer.writerow([label] + landmarks)
                    print(f"Gesture '{label}' saved.")

        # Show the video frame
        cv2.imshow('Gesture Capture', frame)

        # Quit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
hands.close()
