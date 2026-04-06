FROM python:3.12-slim

WORKDIR /app

# Copy the requirements file into the image
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Environment variable for Gemini API Key (Empty by default)
ENV GOOGLE_API_KEY=""

# Hugging Face exposes port 7860
EXPOSE 7860

# Fixed path to the FastAPI app in server/app.py
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]

