# OCR Extraction

A lightweight OCR web app with English, Hindi, and Kannada recognition, interactive word overlays, and searchable PDF exports. The FastAPI backend serves the plain HTML/CSS/JavaScript frontend directly.

## Run locally

Install Python dependencies and the Tesseract engine with the desired language packs:

```bash
python -m pip install -r requirements.txt
```

Install Tesseract system packages (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin tesseract-ocr-kan
```

Then start the complete app with:

```bash
python app.py
```

Open [http://localhost:8000](http://localhost:8000). Set `PORT` to change the listening port. The server binds to `0.0.0.0` for container hosting. If Hindi or Kannada data is unavailable, the UI reports which package is missing; English can still be selected independently.

## Deploy to Render

1. Push this repository to GitHub and create a new **Web Service** in Render from the repository.
2. Select **Docker** as the runtime. Render builds the included Dockerfile, which installs Tesseract and all three language packs.
3. Deploy. The container starts with `python app.py`; Render's `PORT` environment variable is honored automatically.

## API

- `GET /health` — basic liveness check.
- `POST /api/ocr?language=eng` — multipart form upload using the `file` field. Supported language values are `eng`, `hin`, `kan`, `eng+hin`, `eng+kan`, and `eng+hin+kan`. Returns extracted text, word and line bounding boxes, confidence scores, and image dimensions.
- `POST /api/ocr/pdf?language=eng` — same upload format; returns a Tesseract-generated searchable PDF.

Supported input formats are PNG, JPG/JPEG, and WEBP, up to 20 MB and 40 megapixels per image.
