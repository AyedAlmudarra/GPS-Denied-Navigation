#!/bin/bash

echo "Starting setup script for ArduPilot Gazebo..."

# Change directory to home
cd ~

# Update package lists
echo "Updating package lists..."
sudo apt update

# Install necessary dependencies
echo "Installing required libraries for Gazebo and GStreamer..."
sudo apt install -y libgz-sim8-dev rapidjson-dev
sudo apt install -y libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl

# Set GZ_VERSION environment variable
echo "Setting GZ_VERSION environment variable..."
echo 'export GZ_VERSION=harmonic' >> ~/.bashrc

# Apply changes to current session
echo "Applying environment changes..."
source ~/.bashrc

# Install additional dependencies
echo "Installing additional required libraries..."
sudo apt-get install -y libgirepository1.0-dev libcairo2-dev gobject-introspection

# Install Python dependencies
echo "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python dependencies."
    exit 1
fi

echo "Setup complete! You may need to restart your terminal for all changes to take effect."
