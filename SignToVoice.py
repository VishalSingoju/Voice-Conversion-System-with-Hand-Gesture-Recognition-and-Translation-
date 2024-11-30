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
    "ok_sign": "Okay!",
    "peace_sign": "Peace!",
    "open_palm": "Stop!",
    "pointing": "Look there!",
    "call_me": "Call me!",
    "rock": "Rock on!",
    "thumbs_up": "Good job!",
    "thumbs_down": "Not good!",
}

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to recognize gestures
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index tip
    middle_tip = landmarks[12]  # Middle tip
    ring_tip = landmarks[16]  # Ring tip
    pinky_tip = landmarks[20]  # Pinky tip
    wrist = landmarks[0]  # Wrist

# Thumbs Up Gesture
    if (thumb_tip.y < wrist.y and
        all(finger_tip.y > wrist.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return "thumbs_up"

# Thumbs Down Gesture
    if (thumb_tip.y > wrist.y and
        all(finger_tip.y > wrist.y for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])):
        return "thumbs_down"


    # OK Sign Gesture
    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return "ok_sign"

    # Peace Sign Gesture
    if (index_tip.y < landmarks[6].y and
        middle_tip.y < landmarks[10].y and
        ring_tip.y > landmarks[14].y and
        pinky_tip.y > landmarks[18].y):
        return "peace_sign"

    # Open Palm Gesture
    if (all(landmarks[i].y < landmarks[i-2].y for i in [8, 12, 16, 20]) and
        abs(thumb_tip.x - landmarks[17].x) > 0.2):
        return "open_palm"

    # Pointing Gesture
    if (index_tip.y < landmarks[6].y and
        all(landmarks[i].y > landmarks[i-2].y for i in [12, 16, 20]) and
        abs(index_tip.x - wrist.x) > 0.1):
        return "pointing"

    # Call Me Gesture
    if (thumb_tip.y < landmarks[2].y and
        pinky_tip.y < landmarks[18].y and
        all(landmarks[i].y > landmarks[i-2].y for i in [8, 12, 16])):
        return "call_me"

    # Rock Gesture
    if (index_tip.y < landmarks[6].y and
        pinky_tip.y < landmarks[18].y and
        all(landmarks[i].y > landmarks[i-2].y for i in [12, 16])):
        return "rock"

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
