import cv2
import mediapipe as mp
import numpy as np
import os

# ===== CREATE FOLDERS =====
for folder in ["dataset/A", "dataset/B", "dataset/C"]:
    os.makedirs(folder, exist_ok=True)


# ===== AUTO FILE NAME =====
def get_next_filename(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".png")]
    numbers = []

    for f in files:
        try:
            num = int(f.replace("img", "").replace(".png", ""))
            numbers.append(num)
        except:
            pass

    next_num = max(numbers) + 1 if numbers else 1
    return f"{folder}/img{next_num}.png"


# ===== MEDIAPIPE =====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0


# ===== TEXT =====
def draw_text(img, text, pos, scale=0.6, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thickness + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness)


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lm = handLms.landmark

            x = int(lm[8].x * 640)
            y = int(lm[8].y * 480)

            # Draw when finger up
            if lm[8].y < lm[6].y:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 10)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    combined = cv2.add(img, canvas)

    # ===== UI =====
    draw_text(combined, "DATA COLLECTION MODE", (10, 25), 0.7)
    draw_text(combined, "A: Save A | B: Save B | C: Save C", (10, 60), 0.6)
    draw_text(combined, "SPACE: Clear | ESC: Exit", (10, 90), 0.6)

    key = cv2.waitKey(1)

    # ===== SAVE =====
    if key == ord('a'):
        path = get_next_filename("dataset/A")
        cv2.imwrite(path, canvas)
        print("Saved A:", path)

    if key == ord('b'):
        path = get_next_filename("dataset/B")
        cv2.imwrite(path, canvas)
        print("Saved B:", path)

    if key == ord('c'):
        path = get_next_filename("dataset/C")
        cv2.imwrite(path, canvas)
        print("Saved C:", path)

    # ✅ CLEAR WITH SPACE
    if key == 32:
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # EXIT
    if key == 27:
        break

    cv2.imshow("Data Collection - Air Writing", combined)

cap.release()
cv2.destroyAllWindows()