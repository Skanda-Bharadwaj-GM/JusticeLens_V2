# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV/Image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY src/ src/
COPY train_cloud.py .

# Command to run when the container starts
CMD ["python", "train_cloud.py"]