# Nvidia CUDA Base Image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    clang \
    docker.io

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

RUN sudo apt-get install -y nvidia-container-toolkit

# Create the workspace directory
RUN mkdir -p /llama3-workspace

# Set the working directory
WORKDIR /llama3-workspace

COPY . /llama3-workspace

# Default command to keep the container running and allow manual commands
CMD ["/bin/bash"]

# build
# docker build -t llama3-8b-cuda-inference .

# run
# docker run --gpus all -it llama3-8b-cuda-inference /bin/bash