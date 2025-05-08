#!/bin/sh
apt-get update
apt-get install -y tzdata
ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime
dpkg-reconfigure -f noninteractive tzdata


# Install curl and network tools
apt-get install -y curl net-tools iproute2 iputils-ping tailscale

tailscale up --auth-key=tskey-auth-kziQxENWxb11CNTRL-c8i9AirkmDM3BWxvEHKLEMMETG8QRUh8a

# Update alternatives to make Python 3.11 the default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install gcc and git
apt-get update
apt-get install -y build-essential gcc g++ clang git make cmake

git clone https://github.com/CyberDataLab/nebula.git -b physical-deployment
cd ./nebula

curl -fsSL https://astral.sh/uv/install.sh | sh

uv python install 3.11.7

uv python pin 3.11.7

uv sync --group core

source .venv/bin/activate

fastapi run main.py
