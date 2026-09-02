import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection, Hands, dan Drawing
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5
) as face_detection, mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      print("Mengabaikan frame kosong dari kamera.")
      continue

    # Balikkan frame secara horizontal (efek cermin / mirror)
    frame = cv2.flip(frame, 1)

    # Ubah format warna dari BGR ke RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Proses deteksi wajah dan tangan
    results_face = face_detection.process(image)
    results_hands = hands.process(image)

    # Kembalikan format warna ke BGR untuk OpenCV
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w, c = image.shape

    # Deteksi Wajah
    if results_face.detections:
      for detection in results_face.detections:
        mp_drawing.draw_detection(image, detection)
        score = detection.score[0]
        percent = int(score * 100)
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * w)
        y = int(bboxC.ymin * h) - 10
        cv2.putText(image, f"Face: {percent}%", (x, max(y, 20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Deteksi Tangan dan Hitung Jari
    if results_hands.multi_hand_landmarks and results_hands.multi_handedness:
      for idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS
        )
        
        classification = results_hands.multi_handedness[idx]
        score = classification.classification[0].score
        hand_label = classification.classification[0].label # 'Left' atau 'Right'
        percent = int(score * 100)
        
        landmarks = hand_landmarks.landmark
        fingers = []

        # Hitung ukuran referensi tangan (jarak dari pergelangan [0] ke pangkal jari tengah [9])
        hand_size = ((landmarks[0].x - landmarks[9].x)**2 + (landmarks[0].y - landmarks[9].y)**2)**0.5

        # 1. Jempol (Menggunakan jarak antara ujung jempol [4] dan pangkal telunjuk [5])
        thumb_dist = ((landmarks[4].x - landmarks[5].x)**2 + (landmarks[4].y - landmarks[5].y)**2)**0.5
        
        # Jika jarak lebih dari 35% ukuran tangan, artinya jempol terbuka
        if thumb_dist > 0.35 * hand_size:
            fingers.append(1)
        else:
            fingers.append(0)

        # 2. 4 Jari lainnya (Telunjuk, Tengah, Manis, Kelingking)
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]
        
        for tip, pip in zip(tip_ids, pip_ids):
            if landmarks[tip].y < landmarks[pip].y:
                fingers.append(1)  # Jari terbuka
            else:
                fingers.append(0)  # Jari tertutup (genggam)

        total_fingers = sum(fingers)
        
        # Cetak angka ke terminal
        print(f"[{hand_label}] Number = {total_fingers}")

        # Tampilkan teks pada video
        h_x = int(landmarks[0].x * w)
        h_y = int(landmarks[0].y * h) + 20
        text = f"{hand_label}: {total_fingers} ({percent}%)"
        cv2.putText(image, text, (h_x, h_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Tampilkan jendela video
    cv2.imshow("Deteksi Wajah dan Tangan", image)

    if cv2.waitKey(5) & 0xFF == ord("q"):
      break

cap.release()
cv2.destroyAllWindows()