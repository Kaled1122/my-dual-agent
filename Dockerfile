FROM python:3.11-slim

# --- System dependencies (for OCR, PDF, images) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils tesseract-ocr ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# --- ✅ Pass Render's environment variable into the container ---
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# --- Start your Flask app with gunicorn ---
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
