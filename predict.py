import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# The downloaded model.keras file is actually an HDF5 model.
# Keras 3 only reads the new zip format from a .keras name, so load the .h5 copy.
model = load_model("model.h5")

labels = ["A", "B", "C", "D", "E", "F", "G"]
word = ""

mode = "air"
current_fingers = 0

# BGR colors. 1-4 switch the pen while air writing.
colors = [
    ("Blue", (255, 40, 40)),
    ("Red", (0, 0, 255)),
    ("Green", (0, 255, 0)),
    ("Yellow", (0, 255, 255)),
]
color_index = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
last_scores = None
last_msg = ""


# ===== TEXT =====
def draw_text(img, text, pos, scale=0.6, thickness=2, color=(0, 0, 0)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness + 1)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness)


# ===== PREPROCESS =====
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

    thresh = cv2.resize(thresh, (64, 64))
    thresh = thresh / 255.0

    return np.reshape(thresh, (1, 64, 64, 1))


def center_drawing(canvas, target_long=180):
    """Crop the ink and place it at the size and position the model was trained on."""
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 20)
    if len(xs) == 0:
        return None

    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    pad = 10
    height, width = canvas.shape[:2]
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(width, x1 + pad)
    y1 = min(height, y1 + pad)

    crop = canvas[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    scale = target_long / float(max(crop_h, crop_w))
    new_w = max(1, int(round(crop_w * scale)))
    new_h = max(1, int(round(crop_h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(crop, (new_w, new_h), interpolation=interp)

    centered = np.zeros_like(canvas)
    y = max(0, (height - new_h) // 2)
    x = max(0, (width - new_w) // 2)
    paste_h = min(new_h, height - y)
    paste_w = min(new_w, width - x)
    centered[y:y + paste_h, x:x + paste_w] = resized[:paste_h, :paste_w]
    return centered


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
    if not success:
        continue
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    color_name, draw_color = colors[color_index]

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

                    cv2.line(canvas, (prev_x, prev_y), (x, y), draw_color, 10)
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

                draw_text(img, f"Sign: {sign_letter}", (430, 230), 1)

    else:
        prev_x, prev_y = 0, 0

    # ===== SOLID OVERLAY (NO TRANSPARENCY) =====
    combined = img.copy()
    mask = canvas.astype(bool)
    combined[mask] = canvas[mask]

    # ===== UI =====
    draw_text(combined,
              "T:Toggle  P:Predict  C:Clear  R:Reset  SPACE:Space  ESC:Exit",
              (10, 24), 0.48)

    cv2.rectangle(combined, (10, 36), (32, 56), draw_color, -1)
    cv2.rectangle(combined, (10, 36), (32, 56), (255, 255, 255), 1)
    draw_text(combined,
              f"{color_name}    1 Blue   2 Red   3 Green   4 Yellow",
              (40, 52), 0.48, color=draw_color)

    draw_text(combined, f"Mode: {mode.upper()}", (10, 82), 0.7)
    draw_text(combined, "Air letters: A-G", (250, 82), 0.6)
    draw_text(combined, f"Word: {word}", (10, 116), 0.9)

    if last_scores is not None:
        score_text = " ".join(
            f"{labels[i]}{last_scores[i] * 100:.0f}%" for i in range(len(last_scores))
        )
        draw_text(combined, score_text, (10, 148), 0.55)
    if last_msg:
        draw_text(combined, last_msg, (10, 176), 0.55)

    key = cv2.waitKey(1)

    # ===== PEN COLOR =====
    if key in (ord("1"), ord("2"), ord("3"), ord("4")):
        color_index = key - ord("1")
        prev_x, prev_y = 0, 0

    # ===== TOGGLE =====
    if key == ord("t"):
        mode = "sign" if mode == "air" else "air"
        prev_x, prev_y = 0, 0
        print("Switched to", mode)

    # ===== AIR PREDICT =====
    if key == ord("p") and mode == "air":
        normalized = center_drawing(canvas)
        if normalized is None:
            last_scores = None
            last_msg = "Draw a letter first"
        else:
            prediction = model.predict(preprocess(normalized), verbose=0)[0]
            last_scores = prediction
            class_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            letter = labels[class_id]
            print("Prediction:", letter, [round(float(v), 3) for v in prediction])
            if confidence > 0.5:
                word += letter
                last_msg = f"Added {letter}"
            else:
                last_msg = "Not sure, draw again"

    # ===== SIGN PREDICT =====
    if key == ord("p") and mode == "sign":
        if current_fingers == 1:
            word += "A"
            last_msg = "Added A"
        elif current_fingers == 2:
            word += "B"
            last_msg = "Added B"
        elif current_fingers == 3:
            word += "C"
            last_msg = "Added C"
        else:
            last_msg = "Show 1, 2, or 3 fingers"

    # ===== CLEAR =====
    if key == ord("c"):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        prev_x, prev_y = 0, 0

    # ===== RESET =====
    if key == ord("r"):
        word = ""
        last_scores = None
        last_msg = ""

    # ===== SPACE =====
    if key == 32:
        word += " "

    cv2.imshow("Air + Sign AI System", combined)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
