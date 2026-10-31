"""Synthetic air-writing images.

Photos of printed letters do not match this project: training images are
bright strokes on a black 640x480 canvas. These drawings follow that style.
"""

import os

import cv2
import numpy as np

CANVAS_W = 640
CANVAS_H = 480
LETTERS = "DEFG"
PER_LETTER = 50


def line(p0, p1, n=28):
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return p0 * (1 - t) + p1 * t


def arc(cx, cy, rx, ry, a0, a1, n=56):
    ang = np.linspace(a0, a1, n, dtype=np.float32)
    return np.stack(
        [cx + rx * np.cos(ang), cy + ry * np.sin(ang)],
        axis=1,
    ).astype(np.float32)


def strokes_for(letter, rng):
    if letter == "D":
        x = rng.uniform(-74, -64)
        top, bot = -104.0, 104.0
        spine = line((x, top), (x, bot))
        bowl = arc(x, 0, rng.uniform(125, 155), (bot - top) / 2, -np.pi / 2, np.pi / 2)
        bowl[0] = spine[0]
        bowl[-1] = spine[-1]
        return [np.vstack([spine, bowl[1:]])]

    if letter == "E":
        x = rng.uniform(-78, -64)
        top_w = rng.uniform(120, 155)
        mid_w = rng.uniform(70, 105)
        bot_w = rng.uniform(115, 155)
        y_mid = rng.uniform(-8, 10)
        return [
            line((x, -108), (x, 108)),
            line((x, -102), (x + top_w, -102 + rng.uniform(-3, 3))),
            line((x, y_mid), (x + mid_w, y_mid + rng.uniform(-3, 3))),
            line((x, 102), (x + bot_w, 102 + rng.uniform(-3, 3))),
        ]

    if letter == "F":
        x = rng.uniform(-72, -58)
        top_w = rng.uniform(115, 150)
        mid_w = rng.uniform(62, 98)
        y_mid = rng.uniform(-18, 4)
        return [
            line((x, -108), (x, 108)),
            line((x, -102), (x + top_w, -100)),
            line((x, y_mid), (x + mid_w, y_mid)),
        ]

    if letter == "G":
        rx, ry = rng.uniform(72, 92), rng.uniform(96, 118)
        curve = arc(0, 0, rx, ry, -0.55, -2 * np.pi + 0.15)
        end = curve[-1]
        bar_len = rng.uniform(70, 100)
        bar = line(end, (end[0] - bar_len, end[1] + rng.uniform(-4, 4)))
        return [np.vstack([curve, bar[1:]])]

    raise ValueError(letter)


def place(strokes, rng):
    scale = rng.uniform(0.88, 1.12)
    angle = np.deg2rad(rng.uniform(-8, 8))
    shear = rng.uniform(-0.08, 0.08)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    center = np.array(
        [rng.uniform(250, 410), rng.uniform(175, 295)],
        dtype=np.float32,
    )
    placed = []
    for pts in strokes:
        p = pts.copy()
        p[:, 0] += shear * p[:, 1]
        p *= scale
        p = p @ rotation.T
        p += center
        n = len(p)
        t = np.linspace(0, 1, n, dtype=np.float32)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(2.5, 6.5)
        p[:, 0] += amp * np.sin(2 * np.pi * rng.uniform(0.8, 1.8) * t + phase)
        p[:, 1] += amp * np.sin(2 * np.pi * rng.uniform(0.8, 1.8) * t + phase + 1.2)
        p += rng.normal(0, 0.7, p.shape).astype(np.float32)
        placed.append(p)
    return placed


def render(letter, rng):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    color = (0, 255, 0) if rng.random() < 0.7 else (255, 255, 255)
    thickness = int(rng.integers(8, 13))
    for pts in place(strokes_for(letter, rng), rng):
        poly = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, color, thickness, cv2.LINE_AA)
    return canvas


def main():
    rng = np.random.default_rng(31)
    for letter in LETTERS:
        folder = os.path.join("dataset", letter)
        os.makedirs(folder, exist_ok=True)
        for old in os.listdir(folder):
            if old.startswith("gen_") and old.endswith(".png"):
                os.remove(os.path.join(folder, old))
        for i in range(1, PER_LETTER + 1):
            path = os.path.join(folder, f"gen_{i:03d}.png")
            cv2.imwrite(path, render(letter, rng))
        print(f"{letter}: {PER_LETTER} images -> {folder}")


if __name__ == "__main__":
    main()
