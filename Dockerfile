FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends poppler-utils tesseract-ocr ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
