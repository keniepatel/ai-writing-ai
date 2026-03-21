import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("model.keras")

labels = ["A", "B", "C"]
word = ""

mode = "air"
current_fingers = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0


# ===== TEXT =====
def draw_text(img, text, pos, scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness + 1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness)


# ===== PREPROCESS =====
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.resize(thresh, (64, 64))
    thresh = thresh / 255.0

    return np.reshape(thresh, (1, 64, 64, 1))


# ===== FINGER COUNT =====
def count_fingers(lm):
    count = 0
    tips = [8, 12, 16, 20]

    for tip in tips:
        if lm[tip].y < lm[tip - 2].y:
            count += 1

    return count


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark

            # ===== AIR MODE =====
            if mode == "air":
                x = int(lm[8].x * 640)
                y = int(lm[8].y * 480)

                if lm[8].y < lm[6].y:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y

                    # 🔵 DARK BLUE SOLID DRAW
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 40, 40), 10)
                    prev_x, prev_y = x, y
                else:
                    prev_x, prev_y = 0, 0

            # ===== SIGN MODE =====
            else:
                current_fingers = count_fingers(lm)

                if current_fingers == 1:
                    sign_letter = "A"
                elif current_fingers == 2:
                    sign_letter = "B"
                elif current_fingers == 3:
                    sign_letter = "C"
                else:
                    sign_letter = ""

                draw_text(img, f"Sign: {sign_letter}", (400, 100), 1)

    else:
        prev_x, prev_y = 0, 0

    # ===== SOLID OVERLAY (NO TRANSPARENCY) =====
    combined = img.copy()
    mask = canvas.astype(bool)
    combined[mask] = canvas[mask]

    # ===== UI =====
    draw_text(combined,
              "T:Toggle | P:Predict | C:Clear | R:Reset | SPACE:Space | ESC:Exit",
              (10, 25), 0.5)

    draw_text(combined,
              f"Mode: {mode.upper()}",
              (10, 55), 0.7)

    draw_text(combined,
              f"Word: {word}",
              (10, 90), 0.9)

    key = cv2.waitKey(1)

    # ===== TOGGLE =====
    if key == ord('t'):
        mode = "sign" if mode == "air" else "air"
        print("Switched to", mode)

    # ===== AIR PREDICT =====
    if key == ord('p') and mode == "air":
        processed = preprocess(canvas)
        prediction = model.predict(processed)

        class_id = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.5:
            letter = labels[class_id]
            word += letter
            print("Prediction:", letter)

    # ===== SIGN PREDICT =====
    if key == ord('p') and mode == "sign":
        if current_fingers == 1:
            word += "A"
        elif current_fingers == 2:
            word += "B"
        elif current_fingers == 3:
            word += "C"

    # ===== CLEAR =====
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # ===== RESET =====
    if key == ord('r'):
        word = ""

    # ===== SPACE =====
    if key == 32:
        word += " "

    cv2.imshow("Air + Sign AI System", combined)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()