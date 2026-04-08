# Use lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port (7860 is default for Hugging Face Spaces)
EXPOSE 7860

# Command to run the application
# We use uvicorn to host the FastAPI environment
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "7860"]
