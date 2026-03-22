import cv2
import numpy as np
import os
import mediapipe as mp

# SETTINGS
word = "hello"          # change word each time
samples = 30            # number of samples
frames_per_sample = 30  # frames per sample

save_path = "sign_language_dataset/data"

mp_hands = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        keypoints = []
        for lm in hand.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        return keypoints
    return [0]*63  # if no hand detected

for sample in range(samples):
    sequence = []
    print(f"Recording sample {sample+1}")

    for frame_num in range(frames_per_sample):
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        cv2.putText(frame, f"{word} Sample {sample+1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sequence = np.array(sequence)

    filename = f"{word}_{sample+1}.npy"
    np.save(os.path.join(save_path, filename), sequence)

cap.release()
cv2.destroyAllWindows()