import cv2
import os
import time

# Load model deteksi wajah
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Folder penyimpanan foto wajah
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

# Inisialisasi kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera tidak ditemukan!")
    exit()

total_photos = 50
delay_between_photos = 1  # detik
photo_count = 0
last_capture_time = time.time()

print("Mulai ambil wajah dalam 3 detik...")
time.sleep(3)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Gambar kotak di wajah
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Simpan wajah setiap 1 detik
        current_time = time.time()
        if current_time - last_capture_time >= delay_between_photos and photo_count < total_photos:
            face_img = frame[y:y+h, x:x+w]
            filename = os.path.join(save_dir, f"face_{photo_count + 1}.jpg")
            cv2.imwrite(filename, face_img)
            print(f"[{photo_count + 1}/{total_photos}] Wajah disimpan: {filename}")
            photo_count += 1
            last_capture_time = current_time

    cv2.imshow("Deteksi Wajah - Tekan 'q' untuk keluar", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Keluar dari program.")
        break

    if photo_count >= total_photos:
        print("Selesai ambil semua wajah.")
        break

cap.release()
cv2.destroyAllWindows()
