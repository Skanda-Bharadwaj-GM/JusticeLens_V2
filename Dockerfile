# 1. Start with the newer PyTorch 2.4 image to satisfy Hugging Face
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy ALL your project files (This ensures 'src' and 'train_cloud.py' are included)
COPY . .

# 5. Tell the container what to do when it wakes up
CMD ["python", "train_cloud.py"]