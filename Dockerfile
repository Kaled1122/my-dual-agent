# ---------- BASE IMAGE ----------
FROM python:3.11-slim

# ---------- SYSTEM DEPENDENCIES ----------
# poppler-utils  -> for PDF to image conversion (OCR)
# tesseract-ocr  -> OCR engine for images/scanned PDFs
# ffmpeg         -> extract audio from videos
RUN apt-get update && \
    apt-get install -y poppler-utils tesseract-ocr ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- WORKDIR ----------
WORKDIR /app

# ---------- COPY PROJECT ----------
COPY . .

# ---------- PYTHON DEPENDENCIES ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- PORT ----------
EXPOSE 5000

# ---------- START COMMAND ----------
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
