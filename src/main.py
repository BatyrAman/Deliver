# robot_ai_mvp.py
# pip install opencv-python numpy
import time
from pathlib import Path

import cv2
import numpy as np
import s3_client
from config import S3_PREFIX

# CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_alt.xml")
# или если хочешь default:
# CASCADE_PATH = Path(__file__).with_name("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

# face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

if face_cascade.empty():
    raise RuntimeError(f"Не удалось загрузить cascade из файла: {face_cascade}")

# ====== GPIO: включать только на Raspberry Pi ======
USE_GPIO = False  # на Windows оставь False

GPIO = None
pwmA = pwmB = None

if USE_GPIO:
    try:
        import RPi.GPIO as _GPIO
        GPIO = _GPIO

        # Пример пинов (поменяй под свою схему)
        IN1, IN2, ENA = 5, 6, 13      # левый мотор
        IN3, IN4, ENB = 16, 20, 12    # правый мотор

        GPIO.setmode(GPIO.BCM)
        for pin in [IN1, IN2, ENA, IN3, IN4, ENB]:
            GPIO.setup(pin, GPIO.OUT)

        pwmA = GPIO.PWM(ENA, 1000)
        pwmB = GPIO.PWM(ENB, 1000)
        pwmA.start(0)
        pwmB.start(0)

    except Exception as e:
        print("[GPIO] Disabled (import/init failed):", e)
        USE_GPIO = False

# ====== Haar Cascade (берём файл рядом с main.py) ======

if face_cascade.empty():
    raise RuntimeError("Не удалось загрузить Haar Cascade из OpenCV")

if face_cascade.empty():
    raise RuntimeError(f"Не удалось загрузить cascade: {face_cascade}")

def set_motor(left_speed: float, right_speed: float):
    left_speed = max(-1.0, min(1.0, left_speed))
    right_speed = max(-1.0, min(1.0, right_speed))

    if not USE_GPIO:
        print(f"[MOTOR] L={left_speed:+.2f} R={right_speed:+.2f}")
        return

    def drive(in_a, in_b, pwm, sp):
        if sp >= 0:
            GPIO.output(in_a, GPIO.HIGH)
            GPIO.output(in_b, GPIO.LOW)
        else:
            GPIO.output(in_a, GPIO.LOW)
            GPIO.output(in_b, GPIO.HIGH)
        pwm.ChangeDutyCycle(abs(sp) * 100)

    drive(IN1, IN2, pwmA, left_speed)
    drive(IN3, IN4, pwmB, right_speed)

def stop():
    set_motor(0, 0)

# ====== (B) "ИИ": простой детектор цели по цвету (например, красный объект) ======
def detect_target_center(frame_bgr):
    """
    Возвращает (cx, cy, area) для крупнейшего красного пятна, иначе None.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Красный цвет в HSV часто в двух диапазонах
    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # шум уберём
    mask = cv2.medianBlur(mask, 7)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < 800:  # порог — подстрой
        return None

    x, y, w, h = cv2.boundingRect(c)
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy, area
def upload_faces_to_s3(uploader, frame_bgr, s3_prefix: str, ts: int):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return 0

    uploaded = 0
    for i, (x, y, w, h) in enumerate(faces):
        face = frame_bgr[y:y+h, x:x+w]

        ok, buf = cv2.imencode(".jpg", face, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            continue

        key = f"{s3_prefix}faces/face_{ts}_{i}.jpg"
        if uploader.upload_bytes(buf.tobytes(), key, content_type="image/jpeg"):
            uploaded += 1

    return uploaded

def upload_frame_to_s3(uploader, frame_bgr, key: str, jpeg_quality: int = 90):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        print("[S3] JPEG encode failed")
        return False
    return uploader.upload_bytes(buf.tobytes(), key, content_type="image/jpeg")

# ====== (C) ЛОГИКА: ехать к цели, иначе "поиск" ======
def main():
    cap = cv2.VideoCapture(0)  # для Pi Camera через v4l2 тоже часто 0
    if not cap.isOpened():
        raise RuntimeError("Камера не открылась. Проверь /dev/video1 и права доступа.")

    # параметры поведения
    base_speed = 0.35
    turn_gain = 0.006  # как сильно поворачивать по ошибке центра
    search_turn = 0.25

    last_save = 0
    uploader = s3_client.S3Uploader()

    # ---- 1) Пример: загрузка файла с диска ----
    test_file = Path("test.jpg")  # положи рядом любой jpg
    if test_file.exists():
        key = f"{S3_PREFIX}manual_{int(time.time())}.jpg"
        ok = uploader.upload_file(str(test_file), key, content_type="image/jpeg")
        print("upload_file:", ok, key)
    else:
        print("Файл test.jpg не найден — пропускаю upload_file пример.")

    # ---- 2) Пример: загрузка байтов (без файла) ----
    data = b"hello from windows"
    key2 = f"{S3_PREFIX}debug_{int(time.time())}.txt"
    ok2 = uploader.upload_bytes(data, key2, content_type="text/plain")
    print("upload_bytes:", ok2, key2)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            h, w = frame.shape[:2]
            target = detect_target_center(frame)

            now = time.time()

            # 1) Сохраняем ТОЛЬКО лица раз в 2 секунды (независимо от красного объекта)
            if now - last_save > 2.0:
                ts = int(now)
                count = upload_faces_to_s3(uploader, frame, S3_PREFIX, ts)
                if count > 0:
                    print(f"[FACES->S3] uploaded={count}")
                    last_save = now

            # 2) Движение по красному объекту — отдельно (если нужно)

            if target is None:
                set_motor(-search_turn, search_turn)
            else:
                cx, cy, area = target
                error = (cx - (w / 2.0))
                turn = error * turn_gain
                set_motor(base_speed + turn, base_speed - turn)

                # отрисовка (удобно для отладки)
                cv2.circle(frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                cv2.putText(frame, f"area={area:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("robot", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        stop()
        cap.release()
        cv2.destroyAllWindows()
        if USE_GPIO:
            GPIO.cleanup()

if __name__ == "__main__":
    main()
