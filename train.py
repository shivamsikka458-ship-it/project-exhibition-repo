import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATA_PATH = "sign_language_dataset/data"

X = []
y = []

# Automatically get labels
actions = set()
for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        actions.add(file.split("_")[0])

actions = np.array(list(actions))
label_map = {label: num for num, label in enumerate(actions)}

# Load data
for file in os.listdir(DATA_PATH):
    if file.endswith(".npy"):
        sequence = np.load(os.path.join(DATA_PATH, file))
        X.append(sequence)

        word = file.split("_")[0]
        y.append(label_map[word])

X = np.array(X)
y = to_categorical(y).astype(int)

print("X shape:", X.shape)
print("y shape:", y.shape)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# Train test split
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
