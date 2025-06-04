#!/bin/bash

# Azure Kinect SDK Installation Script
# This script downloads and installs the Azure Kinect SDK packages and udev rules

set -e  # Exit on any error

echo "=== Azure Kinect SDK Installation ==="
echo

# Create temporary directory for downloads
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"
cd "$TEMP_DIR"

echo "1. Downloading Azure Kinect SDK .deb packages..."
echo "   - Downloading libk4a1.4-dev..."
wget -q --show-progress https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4-dev/libk4a1.4-dev_1.4.1_amd64.deb

echo "   - Downloading libk4a1.4..."
wget -q --show-progress https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/libk4a1.4/libk4a1.4_1.4.1_amd64.deb

echo
echo "2. Installing .deb packages..."
sudo apt update
sudo apt install -y ./*.deb

echo
echo "3. Downloading udev rules file..."
wget -q --show-progress https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/develop/scripts/99-k4a.rules

echo
echo "4. Installing udev rules..."
sudo cp 99-k4a.rules /etc/udev/rules.d/
sudo chmod 644 /etc/udev/rules.d/99-k4a.rules

echo
echo "5. Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo
echo "6. Cleaning up temporary files..."
cd ~
rm -rf "$TEMP_DIR"

echo
echo "=== Installation Complete! ==="
echo "Azure Kinect SDK has been successfully installed."
echo
echo "Next steps:"
echo "- Reboot your system or reconnect your Azure Kinect device"
echo "- Test the installation with: k4aviewer (if available)"
echo "- Your Azure Kinect device should now be accessible without sudo"
echo