import os
import numpy as np
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

# Define constants
DATA_PATH = "MP_Data"  # Update if needed
actions = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', 'C', ..., 'Z']
no_sequences = 24  # Adjust based on dataset
sequence_length = 30  # Number of frames per sequence

# Map labels
label_map = {label: num for num, label in enumerate(actions)}

# Data collection
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
            if os.path.exists(file_path):
                res = np.load(file_path, allow_pickle=True)
                if res.shape != (63,):  # Ensure each feature vector has 63 values
                    print(f"Skipping {file_path} due to incorrect shape: {res.shape}")
                    continue
                window.append(res)
            else:
                print(f"Missing file: {file_path}")
                continue  # Skip missing frames
        if len(window) == sequence_length:  # Ensure full sequence
            sequences.append(window)
            labels.append(label_map[action])

# Ensure consistent shape before converting to numpy array
if len(sequences) == 0:
    raise ValueError("No valid sequences found. Check your dataset.")

# Convert to NumPy arrays
X = np.array(sequences, dtype=np.float32)  # Ensure all values are float32
y = to_categorical(labels, num_classes=len(actions)).astype(int)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# TensorBoard logs
log_dir = os.path.join("Logs")
tb_callback = TensorBoard(log_dir=log_dir)

# Define Model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation="relu", input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation="relu"))
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(len(actions), activation="softmax"))  # Output layer for 26 classes

# Compile Model
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

# Train Model
model.fit(X_train, y_train, epochs=500, callbacks=[tb_callback])

# Model Summary
model.summary()

# Save Model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save("model.h5")

print("Model training complete and saved successfully.")
