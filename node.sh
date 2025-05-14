#!/bin/bash
set -e  # Exit immediately if any command fails

# Ask for sudo password once at the beginning
sudo -v

################################################################################
# 1. Remove duplicate entries for bookworm-backports if they exist
################################################################################
if [ -f /etc/apt/sources.list.d/backports.list ]; then
  echo "🧹 Removing duplicate bookworm-backports entry..."
  sudo rm /etc/apt/sources.list.d/backports.list
fi

################################################################################
# 2. Add the official Tailscale repository (signed and stable)
################################################################################
curl -fsSL https://pkgs.tailscale.com/stable/debian/bookworm.gpg | \
  sudo tee /usr/share/keyrings/tailscale-archive-keyring.asc >/dev/null

echo "deb [signed-by=/usr/share/keyrings/tailscale-archive-keyring.asc] https://pkgs.tailscale.com/stable/debian bookworm main" | \
  sudo tee /etc/apt/sources.list.d/tailscale.list

################################################################################
# 3. Update and install essential packages and dependencies
################################################################################
sudo apt-get update
sudo apt-get install -y \
  tzdata curl net-tools iproute2 iputils-ping \
  build-essential gcc g++ clang git make cmake \
  tailscale \
  python3.11 python3.11-venv

################################################################################
# 4. Set system timezone to Europe/Madrid
################################################################################
sudo ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime
sudo dpkg-reconfigure -f noninteractive tzdata

################################################################################
# 5. Connect to Tailscale VPN using a valid auth key (it lasts 90 days)
################################################################################
# 🔐 IMPORTANT: Replace this with a valid reusable auth key
sudo tailscale up --auth-key=tskey-auth-k26BCauJup11CNTRL-Ytj6n6t6dNCK7nhdDhY4NC8VvsBC2Xvc
sudo systemctl enable tailscaled

################################################################################
# 6. Set Python 3.11 as the default system Python
################################################################################
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1

################################################################################
# 7. Clone the NEBULA repository if it doesn't already exist
################################################################################
if [ ! -d "nebula" ]; then
  git clone https://github.com/CyberDataLab/nebula.git -b physical-deployment
fi

cd nebula
sudo chown -R "$(whoami)":"$(whoami)" .

################################################################################
# 8. Install UV and create a Python 3.11.7 virtual environment
################################################################################
curl -fsSL https://astral.sh/uv/install.sh | sh
uv python install 3.11.7
uv python pin 3.11.7
uv sync --group core

################################################################################
# 9. Activate the virtual environment and run the FastAPI backend
################################################################################
source .venv/bin/activate
fastapi run main.py
