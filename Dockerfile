FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Standard Secret Configuration for OpenEnv Benchmark
ENV HF_TOKEN=""
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Hugging Face exposes port 7860
EXPOSE 7860

# Fixed path to the FastAPI app in server/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

