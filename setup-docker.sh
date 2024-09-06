#!/bin/bash

# Exit on any error except for the groupadd error when the group already exists
set -e

# Install Nvidia Container Toolkit to allow Docker to access GPUs
echo "Installing Nvidia Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Add the current user to the docker group
echo "Setting up Docker permissions..."
if getent group docker > /dev/null 2>&1; then
    echo "Group 'docker' already exists, skipping group creation."
else
    sudo groupadd docker
fi

sudo usermod -aG docker $USER

# Create a new group for docker (so changes take effect immediately)
newgrp docker

# Test Docker installation
echo "Testing Docker installation with hello-world container..."
docker run hello-world

# Build Docker image and run it with GPU access
echo "Building Docker image..."
docker build -t llama3-8b-cuda-inference .

echo "Running Docker container with GPU access..."
docker run --gpus all -it llama3-8b-cuda-inference /bin/bash
