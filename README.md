# WasteWise AI — Phase 1

WasteWise AI is an end-to-end image-classification project that predicts one of five recyclable waste categories:

- cardboard
- glass
- metal
- paper
- plastic

Phase 1 contains:

1. Dataset validation and stratified train/validation/test splitting
2. MobileNetV2 Transfer Learning Image Classifier
3. Data augmentation and training callbacks
4. Accuracy/loss plots, confusion matrix, and classification report
5. Local single-image prediction
6. FastAPI image-upload prediction API
7. React + Material UI web interface
8. Confidence-based `uncertain` response
9. Optional Docker setup

## Recommended environment

- Python 3.10 or 3.11
- Node.js 20 or newer
- Git
- 8 GB RAM or more
- GPU is optional

## Dataset

For the first version, use the TrashNet dataset. It contains folders for:

- cardboard
- glass
- metal
- paper
- plastic
- trash

Phase 1 intentionally uses only the first five classes.

Dataset source:

https://github.com/garythung/trashnet

After downloading and extracting the dataset, copy the five class folders into:

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
└── plastic/
```

Do not place all images directly inside `data/raw`. Each category must have its own folder.

## Project structure

```text
wastewise-ai/
├── artifacts/                  # Generated model and evaluation files
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   └── model_service.py
│   ├── Dockerfile
│   └── tests/test_api.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── package.json
├── notebooks/
│   └── 01_training_walkthrough.ipynb
├── training/
│   ├── prepare_dataset.py
│   ├── train.py
│   └── predict_local.py
├── docker-compose.yml
├── requirements.txt
└── README.md
```

# Step 1 — Set up Python

From the project root:

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## macOS/Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Verify TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

A GPU is not required. The project will train on CPU, although it will take longer.

# Step 2 — Prepare the dataset

Validate and split the raw images:

```bash
python training/prepare_dataset.py
```

Default split:

- 80% training
- 10% validation
- 10% testing

The script creates:

```text
data/processed/
├── train/
├── val/
└── test/
```

To rebuild the split:

```bash
python training/prepare_dataset.py --overwrite
```

# Step 3 — Train the CNN

```bash
python training/train.py
```

Useful options:

```bash
python training/train.py --epochs 15 --batch-size 32 --image-size 224
```

Generated artifacts:

```text
artifacts/
├── best_model.keras
├── labels.json
├── metrics.json
├── history.csv
├── training_curves.png
├── confusion_matrix.png
└── classification_report.csv
```

# Step 4 — Test a local image

```bash
python training/predict_local.py path/to/test-image.jpg
```

Example output:

```json
{
  "status": "success",
  "predicted_class": "plastic",
  "confidence": 0.8742,
  "top_predictions": [
    {"class_name": "plastic", "confidence": 0.8742},
    {"class_name": "glass", "confidence": 0.0711},
    {"class_name": "metal", "confidence": 0.0314}
  ]
}
```

Predictions below the configured threshold are returned with the status `uncertain`.

# Step 5 — Run the backend

The trained model must exist at `artifacts/best_model.keras`.

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Open the API documentation:

```text
http://localhost:8000/docs
```

Health endpoint:

```text
http://localhost:8000/health
```

# Step 6 — Run the frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

The frontend sends the selected image to:

```text
http://localhost:8000/predict
```

To use another backend URL, create `frontend/.env`:

```env
VITE_API_URL=http://localhost:8000
```

# Optional Docker run

Train the model locally first. Then run:

```bash
docker compose up --build
```

Frontend:

```text
http://localhost:5173
```

Backend docs:

```text
http://localhost:8000/docs
```

# API response format

Successful confident prediction:

```json
{
  "status": "success",
  "predicted_class": "paper",
  "confidence": 0.91,
  "recommendation": "Keep the paper clean and dry before recycling.",
  "top_predictions": []
}
```

Low-confidence prediction:

```json
{
  "status": "uncertain",
  "predicted_class": null,
  "confidence": 0.42,
  "recommendation": "Try another photo with one object, better lighting, and a plain background.",
  "top_predictions": []
}
```
