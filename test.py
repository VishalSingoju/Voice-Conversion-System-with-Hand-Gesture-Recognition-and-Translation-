import cv2
import mediapipe as mp
import pyttsx3
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


# Initialize MediaPipe and pyttsx3
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# Predefined gestures
gesture_dict = {
    "thumbs_up": "Good job!",
    "thumbs_down": "Not good!",
    "ok_sign": "Okay!"
}

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize simple gestures
def recognize_gesture(landmarks):
    # Example gesture recognition logic
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index tip
    pinky_tip = landmarks[20]  # Pinky tip
    wrist = landmarks[0]  # Wrist

    # Thumbs Up Gesture
    if thumb_tip.y < wrist.y and index_tip.y > wrist.y:
        return "thumbs_up"

    # Thumbs Down Gesture
    if thumb_tip.y > wrist.y and index_tip.y > wrist.y:
        return "thumbs_down"

    # OK Sign Gesture
    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "ok_sign"

    return None

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame...")
            continue

        # Flip frame for a mirror-like effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        # Draw hand landmarks and recognize gestures
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract gesture
                landmarks = hand_landmarks.landmark
                gesture = recognize_gesture(landmarks)
                if gesture:
                    text = gesture_dict.get(gesture, "Unknown gesture")
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    speak(text)

        # Display the frame
        cv2.imshow("SignSpeak", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
