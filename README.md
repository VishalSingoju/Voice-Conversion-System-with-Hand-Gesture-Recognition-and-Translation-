# Voice-Conversion-System-with-Hand-Gesture-Recognition-and-Translation-
# SignSpeak - Real-Time Gesture Recognition with MediaPipe

SignSpeak is a Python-based real-time hand gesture recognition system using MediaPipe. This application interprets predefined hand gestures and provides voice feedback for the recognized gestures.

## Features

- **Real-Time Gesture Recognition**: Recognizes gestures using a webcam. it uses "OpenCV2"
- **Predefined Gestures**:
    "ok_sign": "Okay!",
    "peace_sign": "Peace!",
    "open_palm": "Stop!",
    "pointing": "Look there!",
    "call_me": "Call me!",
- **Voice Feedback**: Converts recognized gestures into voice responses using `pyttsx3`.

About MediaPipe
MediaPipe is an open-source framework developed by Google for building multimodal machine learning pipelines. It is designed for real-time processing of data from various sensors and multimedia inputs like video, audio, and text. MediaPipe provides robust pre-trained models and easy-to-use APIs for implementing machine learning solutions on mobile, web, and desktop platforms.