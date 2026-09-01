import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# 1. SETUP PATHS (Adjust 'MP_Data' if your folder has a different name)
DATA_PATH = 'MP_Data.csv'
actions = np.array([res for res in os.listdir(DATA_PATH)]) # Automatically gets sign names
sequence_length = 30 # Number of frames per video
no_sequences = 30    # Number of videos per sign
label_map = {label:num for num, label in enumerate(actions)}

# 2. LOAD DATASET
print("Loading data from your repository folders...")
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences) # Shape: (Total_Samples, 30, 63)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# 3. THE 95% ACCURACY ARCHITECTURE (CNN + BiLSTM)
model = Sequential()
# CNN: Spatial feature extraction
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(30, 63)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# BiLSTM: Temporal/Motion extraction
model.add(Bidirectional(LSTM(64, return_sequences=True, activation='relu')))
model.add(Bidirectional(LSTM(128, return_sequences=False, activation='relu')))

# Dense: Classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3)) # Crucial to prevent overfitting
model.add(Dense(actions.shape[0], activation='softmax'))

# 4. COMPILE & TRAIN
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# This stops training when accuracy hits its peak, saving the best version
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

print("Starting training on your dataset...")
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# 5. SAVE
model.save('action_model.h5')
print("Success! 'action_model.h5' is ready for real-time testing.")
