import cv2
import mediapipe as mp
import pyttsx3
import os

# Disable TensorFlow optimizations for compatibility
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define gestures and their corresponding messages
gesture_dict = {
    "thumbs_up": "Good job!",
    "thumbs_down": "Not good!",
    "ok_sign": "Okay!",
    "peace_sign": "Peace!",
    "open_palm": "Stop!",
    "pointing": "Look there!",
    "call_me": "Call me!",
    "rock": "Rock on!",
}

def speak(text):
    """Speak the provided text using pyttsx3."""
    engine.say(text)
    engine.runAndWait()

def recognize_gesture(landmarks):
    """
    Recognize gestures based on hand landmarks.
    Returns a gesture name if recognized; otherwise, None.
    """
    # Extract specific landmarks for gesture detection
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    wrist = landmarks[0]

    # Define gesture logic
    if (thumb_tip.y < wrist.y and
        all(finger_tip.y > wrist.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return "thumbs_up"

    if (thumb_tip.y > wrist.y and
        all(finger_tip.y > wrist.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return "thumbs_down"

    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "ok_sign"

    if (index_tip.y < landmarks[6].y and
        middle_tip.y < landmarks[10].y and
        ring_tip.y > landmarks[14].y and
        pinky_tip.y > landmarks[18].y):
        return "peace_sign"

    if (all(landmarks[i].y < landmarks[i - 2].y for i in [8, 12, 16, 20]) and
        abs(thumb_tip.x - landmarks[17].x) > 0.2):
        return "open_palm"

    if (index_tip.y < landmarks[6].y and
        all(landmarks[i].y > landmarks[i - 2].y for i in [12, 16, 20]) and
        abs(index_tip.x - wrist.x) > 0.1):
        return "pointing"

    if (thumb_tip.y < landmarks[2].y and
        pinky_tip.y < landmarks[18].y and
        all(landmarks[i].y > landmarks[i - 2].y for i in [8, 12, 16])):
        return "call_me"

    if (index_tip.y < landmarks[6].y and
        pinky_tip.y < landmarks[18].y and
        all(landmarks[i].y > landmarks[i - 2].y for i in [12, 16])):
        return "rock"

    return None

# Open webcam for gesture recognition
cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame. Skipping...")
            continue

        # Flip and convert frame to RGB for processing
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Recognize gesture
                landmarks = hand_landmarks.landmark
                gesture = recognize_gesture(landmarks)
                if gesture:
                    message = gesture_dict.get(gesture, "Unknown gesture")
                    cv2.putText(frame, message, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    speak(message)

        # Show the processed video feed
        cv2.imshow("SignSpeak", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
