# Use the official Python image as a base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    pkg-config \
    git && \
    rm -rf /var/lib/apt/lists/*


# Copy and install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training script into the container
COPY test_train.py .

# Run the training script and then start the MLflow UI
CMD ["sh", "-c", "python test_train.py && mlflow ui --host 0.0.0.0 --port 5001"]


## Set the command to run the script
## CMD ["python", "test_train.py"]
## Copy your application code into the container (optional based on your project structure)
## COPY . .

## Specify the command to run your application (example: python app.py)
## CMD ["python", "app.py"]
## Run the MLflow UI as the main entrypoint
## CMD mlflow run test_train.py && mlflow ui --host 0.0.0.0 --port 5001
