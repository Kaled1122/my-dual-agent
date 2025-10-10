FROM python:3.11-slim

# --- System dependencies for OCR, PDF, video ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        ffmpeg \
        libsm6 \
        libxext6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- App setup ---
WORKDIR /app
COPY . .

# --- Install Python deps ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir moviepy imageio[ffmpeg]  # <---- force install manually

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
