import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Optional: menghilangkan warning TensorFlow

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load model CNN yang sudah dilatih (harus RGB input shape: 24x48x3)
model = load_model("eye_state_model.h5")

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark index untuk mata kiri dan kanan (MediaPipe)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# Fungsi ekstrak dan resize gambar mata
def get_eye_image(image, landmarks, eye_indices):
    h, w, _ = image.shape
    eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    x_min = min(p[0] for p in eye_points)
    x_max = max(p[0] for p in eye_points)
    y_min = min(p[1] for p in eye_points)
    y_max = max(p[1] for p in eye_points)

    margin = 5
    x_min = max(x_min - margin, 0)
    x_max = min(x_max + margin, w)
    y_min = max(y_min - margin, 0)
    y_max = min(y_max + margin, h)

    eye_img = image[y_min:y_max, x_min:x_max]
    if eye_img.size == 0:
        return np.zeros((24, 24, 3), dtype=np.uint8)

    return cv2.resize(eye_img, (24, 24))

# Threshold & counter
closed_eyes_frame = 0
drowsy_threshold = 30  # Jumlah frame berturut-turut mata tertutup

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        left_eye_img = get_eye_image(frame, landmarks, LEFT_EYE_IDX)
        right_eye_img = get_eye_image(frame, landmarks, RIGHT_EYE_IDX)

        # Gabungkan kedua mata secara horizontal
        eye_pair = np.hstack((left_eye_img, right_eye_img))  # shape: (24, 48, 3)
        input_img = eye_pair / 255.0  # Normalisasi
        input_img = np.expand_dims(input_img, axis=0)  # shape: (1, 24, 48, 3)

        prediction = model.predict(input_img, verbose=0)[0][0]

        if prediction < 0.5:  # Tertutup
            closed_eyes_frame += 1
        else:
            closed_eyes_frame = 0

        if closed_eyes_frame >= drowsy_threshold:
            cv2.putText(frame, "DROWSINESS ALERT!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    else:
        closed_eyes_frame = 0

    cv2.imshow("Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
