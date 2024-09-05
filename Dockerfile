# Nvidia CUDA Base Image
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Update package list and install necessary packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    clang

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