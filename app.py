import io
import os
from collections import defaultdict
from pathlib import Path

import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from pytesseract import Output, TesseractError, TesseractNotFoundError
from starlette.concurrency import run_in_threadpool


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_LANGUAGES = {"eng", "hin", "kan", "eng+hin", "eng+kan", "eng+hin+kan"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="OCR Extraction", version="1.0.0")


def _read_image(data: bytes, filename: str) -> Image.Image:
    if Path(filename or "").suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PNG, JPG, JPEG, or WEBP image.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 20 MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as source:
            if source.format not in {"PNG", "JPEG", "WEBP"}:
                raise HTTPException(status_code=415, detail="The file is not a supported image.")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions exceed the 40-megapixel limit.",
                )
            image = ImageOps.exif_transpose(source)
            image.load()
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            return image
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded image could not be decoded.") from exc


def _ensure_languages(language: str) -> None:
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=422, detail="Choose a supported OCR language.")
    try:
        installed = set(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Tesseract OCR is not installed. Install Tesseract and its language data, then restart the app.",
        ) from exc
    missing = [part for part in language.split("+") if part not in installed]
    if missing:
        package_hint = ", ".join(f"tesseract-ocr-{code}" for code in missing)
        raise HTTPException(
            status_code=503,
            detail=(
                f"OCR language data is missing for: {', '.join(missing)}. "
                f"Install {package_hint} and try again. English ('eng') remains available if installed."
            ),
        )


async def _upload_bytes(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    await file.close()
    return data


def _extract_ocr(image: Image.Image, language: str) -> dict:
    data = pytesseract.image_to_data(
        image, lang=language, output_type=Output.DICT, config="--psm 3"
    )
    words = []
    grouped = defaultdict(list)

    for index, raw_text in enumerate(data["text"]):
        text = raw_text.strip()
        try:
            confidence = float(data["conf"][index])
        except (ValueError, TypeError):
            confidence = -1.0
        if not text or confidence < 0:
            continue
        word = {
            "left": int(data["left"][index]),
            "top": int(data["top"][index]),
            "width": int(data["width"][index]),
            "height": int(data["height"][index]),
            "text": text,
            "conf": round(confidence, 2),
        }
        words.append(word)
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped[key].append(word)

    lines = []
    paragraphs = []
    previous_paragraph = None
    for key, line_words in sorted(grouped.items()):
        left = min(word["left"] for word in line_words)
        top = min(word["top"] for word in line_words)
        right = max(word["left"] + word["width"] for word in line_words)
        bottom = max(word["top"] + word["height"] for word in line_words)
        line = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
            "text": " ".join(word["text"] for word in line_words),
            "conf": round(sum(word["conf"] for word in line_words) / len(line_words), 2),
        }
        lines.append(line)
        paragraph_key = key[:2]
        if paragraphs and paragraph_key != previous_paragraph:
            paragraphs.append("")
        paragraphs.append(line["text"])
        previous_paragraph = paragraph_key

    return {
        "text": "\n".join(paragraphs),
        "words": words,
        "lines": lines,
        "img_width": image.width,
        "img_height": image.height,
        "language": language,
    }


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/ocr")
async def recognize_text(file: UploadFile = File(...), language: str = "eng"):
    language = language.replace(" ", "+")
    data = await _upload_bytes(file)
    image = await run_in_threadpool(_read_image, data, file.filename or "")
    await run_in_threadpool(_ensure_languages, language)
    try:
        return await run_in_threadpool(_extract_ocr, image, language)
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract OCR is not installed.") from exc
    except TesseractError as exc:
        raise HTTPException(status_code=422, detail=f"Tesseract could not process this image: {exc}") from exc


@app.post("/api/ocr/pdf")
async def searchable_pdf(file: UploadFile = File(...), language: str = "eng"):
    language = language.replace(" ", "+")
    data = await _upload_bytes(file)
    image = await run_in_threadpool(_read_image, data, file.filename or "")
    await run_in_threadpool(_ensure_languages, language)
    try:
        pdf_bytes = await run_in_threadpool(
            pytesseract.image_to_pdf_or_hocr, image, extension="pdf", lang=language
        )
    except TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract OCR is not installed.") from exc
    except TesseractError as exc:
        raise HTTPException(status_code=422, detail=f"Tesseract could not create the PDF: {exc}") from exc
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="searchable-ocr.pdf"'},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
