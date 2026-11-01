# Air Writing & Hand Gesture Recognition

Real-time system that turns an index-finger stroke in the air into letters A–G, and maps a simple finger count to A, B, or C.

It uses a webcam, MediaPipe hand tracking, and a small Keras CNN.

## Features

- Air writing with the index finger for letters A–G
- Pen colors: blue, red, green, and yellow
- Sign mode: 1, 2, or 3 raised fingers become A, B, or C
- Word building, with a confidence check before a letter is added
- CNN trained on grayscale letter images

## Technologies

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy

## Project structure

```
air-writing-ai/
├── dataset/
│   ├── A/ … G/
├── generate_letters.py
├── main.py
├── train_model.py
├── predict.py
├── model.h5
├── model.keras
├── requirements.txt
└── LICENSE
```

`predict.py` loads `model.h5`. `train_model.py` writes both `model.h5` and `model.keras`.

## Installation

```bash
git clone https://github.com/keniepatel/ai-writing-ai.git
cd ai-writing-ai
pip install -r requirements.txt
```

A webcam is required for collection and prediction.

## Usage

Collect letter images. Press A–G to save the current canvas into that letter’s folder. Space clears the canvas. Esc exits.

```bash
python main.py
```

Train the CNN on `dataset/`. This overwrites `model.h5` and `model.keras`.

```bash
python train_model.py
```

Run recognition.

```bash
python predict.py
```

`generate_letters.py` draws extra D–G samples in the same bright-stroke style as the collected images:

```bash
python generate_letters.py
```

## Controls

Data collection (`main.py`):

| Key | Action |
| --- | --- |
| A–G | Save the drawing as that letter |
| Space | Clear the canvas |
| Esc | Exit |

Recognition (`predict.py`):

| Key | Action |
| --- | --- |
| T | Toggle air mode and sign mode |
| P | Predict and append a letter when confident |
| C | Clear the drawing |
| R | Reset the word |
| Space | Add a space |
| 1–4 | Pen color: blue, red, green, yellow |
| Esc | Exit |

In sign mode, show 1, 2, or 3 fingers, then press P.

## How it works

MediaPipe finds one hand. In air mode the index fingertip draws on a black canvas. Before prediction, the ink is cropped, scaled, and centered so it matches the training images. A grayscale CNN then scores letters A–G. A letter is added to the word only when the top score is above 0.5.

## Author

Keni Patel
