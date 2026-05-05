# Face Liveness Detection

A face liveness detection system that asks the user to perform a sequence of facial **actions** (smile, raise eyebrows, head movement, etc.) on camera and verifies them in real time using **MediaPipe FaceMesh** blendshapes and a custom-trained neural network classifier.

This was built around the approach used in my **Rackathon 2025** winning project — combining MediaPipe blendshape thresholds for fast in-browser checks with an offline-trained MLP classifier on facial-landmark coordinates as a stronger verifier.

## What's in this repo

The repository is the data-and-model side of the system. It contains:

- **`Liveness/`** — the in-browser liveness checks (MediaPipe FaceMesh blendshapes via Tasks Vision)
  - `liveness-3-actions.html` — randomly picks 3 actions from the action list
  - `Liveness-6-actions.html` — picks from a 6-action list (smile, head move, raise eyebrows, etc.)
  - `dataset_create.ipynb` — utilities used while collecting the training dataset
  - `mlp.ipynb` — Keras MLP classifier trained on facial-landmark features
- **`Preprocessing/`** — scripts that built the training spreadsheets
  - `data_crop.py` — center-crops every image in a folder to 640×324 (PIL)
  - `landmarksToExcel.py` — runs MediaPipe FaceMesh on each image and exports the 468 landmark `(x, y)` pairs per pose into Excel
  - `Blendshape_score_to_excel.ipynb` — same idea for blendshape category scores
- **`Data/`** — the resulting spreadsheets used by the training notebook
  - `face_landmarks_xy.xlsx` (~39 MB) — landmark coordinates per pose / image
  - `blendshapes_scores.xlsx` (~700 KB) — blendshape category scores per pose

## How it works

### 1. In-browser liveness check (`Liveness/*.html`)

Uses [MediaPipe Tasks Vision FaceLandmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker) configured with `outputFaceBlendshapes: true`. A random subset of actions from the action list is shown to the user one at a time; each action has:

```js
{ name: "Smile",          index: 44, threshold: 0.5 }
{ name: "Raise Eyebrows", indices: [4,5], threshold: 0.7 }
{ name: "Move your head", indices: [12,13], threshold: 0.6 }
// ... and more in the 6-action variant
```

Each frame's blendshape scores are read from `categories[index].score` and compared against the threshold. If the user performs the action within a time window the check advances to the next action.

### 2. Offline MLP classifier (`Liveness/mlp.ipynb`)

A Keras `Sequential` MLP trained on the landmark features in `face_landmarks_xy.xlsx`:

- Features: per-image `(x, y)` coordinates of all 468 MediaPipe FaceMesh landmarks.
- Pipeline: `LabelEncoder` for action labels → `StandardScaler` for features → `train_test_split` (80/20) → MLP with `Dense + Dropout` → `Adam` optimizer → 50 epochs / batch size 32.
- Output: a saved classifier that can verify which action a face is performing without relying solely on hand-tuned blendshape thresholds.

### 3. Data preprocessing (`Preprocessing/`)

- Run `data_crop.py` on each pose folder to standardize image size.
- Run `landmarksToExcel.py` over the cropped folders to produce `face_landmarks_xy.xlsx`.
- Run `Blendshape_score_to_excel.ipynb` to produce `blendshapes_scores.xlsx`.

## Stack

| Layer | Technology |
|-------|-----------|
| In-browser detection | MediaPipe Tasks Vision (FaceLandmarker), HTML/JS |
| Landmark extraction (offline) | MediaPipe Python (`face_mesh`), OpenCV |
| Data prep | Pillow, pandas |
| Model | TensorFlow / Keras MLP + scikit-learn (StandardScaler, LabelEncoder) |

## How to run

### In-browser demo

Open `Liveness/liveness-3-actions.html` or `Liveness-6-actions.html` in a browser with camera permission. The MediaPipe model loads from CDN — no build step required.

### Train your own classifier

1. Collect images per action into folders.
2. Run `Preprocessing/data_crop.py` (edit `main_folder` path inside the script).
3. Run `Preprocessing/landmarksToExcel.py` to generate the feature spreadsheet.
4. Open `Liveness/mlp.ipynb` and run all cells — it reads the spreadsheet and trains the MLP.

## Notes

- The HTML files reference MediaPipe blendshape **indices** (e.g. 44 for `mouthSmileLeft/Right`). If the MediaPipe blendshape category list changes upstream, those indices may need updating.
- The `landmarksToExcel.py` and `data_crop.py` scripts have local Windows paths hardcoded near the bottom — change them before running.
- This is the original SIH/Rackathon-era prototype data + model; for a more polished real-time liveness demo, see [Real-Time-Face-Liveness-Detection-using-Random-Facial-Actions](https://github.com/rupeshbharambe24/Real-Time-Face-Liveness-Detection-using-Random-Facial-Actions).
