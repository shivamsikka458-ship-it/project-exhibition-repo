import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Setup Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Load the trained model
model = load_model('action_model.h5')
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence = [] # Buffer to store 30 frames
threshold = 0.8 # Only show prediction if confidence > 80%

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks for current frame
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            sequence.append(landmarks)
            sequence = sequence[-30:] # Keep only last 30 frames

            # Predict when buffer is full
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                
                # Visual logic
                if res[np.argmax(res)] > threshold:
                    predicted_sign = actions[np.argmax(res)]
                    cv2.putText(image, f"{predicted_sign} ({res[np.argmax(res)]*100:.1f}%)", 
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Detection Live', image)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()