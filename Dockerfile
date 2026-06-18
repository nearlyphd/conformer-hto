# Use the official TensorFlow GPU image with Jupyter pre-installed
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set directory inside the container
WORKDIR /tf

# Install additional system dependencies
USER root
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd /root/.ssh && \
    chmod 700 /root/.ssh

COPY runpod.pub /root/.ssh/authorized_keys
RUN chmod 600 /root/.ssh/authorized_keys && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#AuthorizedKeysFile/AuthorizedKeysFile/' /etc/ssh/sshd_config

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch with CUDA 12.1 (compatible with driver CUDA 12.4)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install timm

# Install Data Science & Visualization libraries
RUN pip install \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    tqdm \
    jupyterlab

RUN pip install ultralytics
RUN pip install opencv-python-headless

# Copy X-ray images into the container
COPY data/hto/xrays/ /tf/data/hto/xrays/

# Copy Conformer library
COPY notebooks/CKD/ /tf/notebooks/CKD/

# Expose ports
EXPOSE 8888
EXPOSE 22

CMD ["bash", "-c", "service ssh start && source /etc/bash.bashrc && jupyter lab \
    --notebook-dir=/tf \
    --ip=0.0.0.0 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.allow_remote_access=True \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.tornado_settings=\"{'headers':{'Content-Security-Policy':'frame-ancestors * self'}}\""]
