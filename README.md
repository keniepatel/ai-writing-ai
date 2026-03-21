# ✍️ Air Writing & Hand Gesture Recognition System

## 📌 Overview

This project is a real-time AI system that allows users to:

* Write letters in the air using hand gestures
* Recognize hand signs (finger counting)
* Convert gestures into text

It uses **Computer Vision + Deep Learning** to provide an interactive experience.

## 🌐 Demo Video


https://github.com/user-attachments/assets/33e17676-f066-4058-b1f7-6b14e56c523f



---

## 🚀 Features

* ✍️ Air writing using index finger
* ✋ Hand sign recognition (1,2,3 fingers → A,B,C)
* 🔄 Toggle between Air Mode and Sign Mode
* 🔤 Word formation
* 🎨 Real-time drawing interface
* 🧠 CNN-based character recognition

---

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* TensorFlow / Keras
* NumPy

---

## 📂 Project Structure

```
air-writing-ai/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│
├── model.keras
├── main.py
├── train_model.py
├── predict.py
├── requirements.txt
```

---

## ⚙️ Installation

```bash
git clone https://github.com/YOUR_USERNAME/air-writing-ai.git
cd air-writing-ai

pip install -r requirements.txt
```

---

## ▶️ Usage

### 1️⃣ Collect Data

```bash
python main.py
```

### 2️⃣ Train Model

```bash
python train_model.py
```

### 3️⃣ Run Application

```bash
python predict.py
```

---

## 🎮 Controls

| Key   | Action        |
| ----- | ------------- |
| T     | Toggle Mode   |
| P     | Predict       |
| C     | Clear Drawing |
| R     | Reset Word    |
| SPACE | Add Space     |
| ESC   | Exit          |

---

## 🧠 How It Works

* MediaPipe detects hand landmarks
* Index finger is used to draw in air
* Drawing is processed and fed into CNN model
* Model predicts characters (A, B, C)
* Finger counting is used for sign recognition

---

## 📈 Future Improvements

* Add full A–Z recognition
* Improve model accuracy
* Add GUI interface
* Add speech output

---

## 👨‍💻 Author

Rahul Prasad

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
