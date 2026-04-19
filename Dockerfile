# Use the official TensorFlow GPU image with Jupyter pre-installed
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set directory inside the container
WORKDIR /tf

# Install additional system dependencies if needed (e.g., for OpenCV or plotting)
USER root
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install PyTorch (Compatible with CUDA 11/12)
# Note: TensorFlow docker images usually come with specific CUDA versions.
# We install the standard PyTorch pip package which includes its own CUDA runtime
# to ensure maximum compatibility.
RUN pip install timm torch torchvision torchaudio

# We force protobuf to be < 6.0.0 to make Tensorflow happy
RUN pip install "protobuf<6.0.0"

# Install Data Science & Visualization libraries
RUN pip install \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    tqdm \
    jupyterlab

# Expose Jupyter port
EXPOSE 8888

# The base image already has an ENTRYPOINT for Jupyter,
# but we can override CMD to ensure custom flags if needed.
# This keeps the default behavior of the tensorflow-jupyter image.
CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/tf --ip 0.0.0.0 --no-browser --allow-root"]
