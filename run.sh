#!/bin/bash
# -----------------------------------------------------------------------------
#  Nebula edge-node bootstrap script
#  – installs required packages
#  – joins the Tailscale tailnet
#  – creates the Python 3.11 virtual-env with “uv”
#  – starts the FastAPI backend on 127.0.0.1
#  – publishes it to the tailnet on port 8000 only
# -----------------------------------------------------------------------------

set -e   # abort the script on any command that exits with a non-zero status

###############################################################################
# 0.  BASE OS PREPARATION                                                     #
###############################################################################
apt-get update

# --- Time-zone ---------------------------------------------------------------
DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
ln -fs /usr/share/zoneinfo/Europe/Madrid /etc/localtime
dpkg-reconfigure -f noninteractive tzdata

###############################################################################
# 1.  NETWORKING TOOLS + TAILSCALE                                             #
###############################################################################
apt-get install -y curl net-tools iproute2 iputils-ping tailscale

# --- Join the tailnet (replace the key with **your own reusable auth-key**) --
tailscale up --auth-key=tskey-REPLACE-ME --hostname=$(hostname) --ssh

###############################################################################
# 2.  COMPILERS & GIT (for Python wheels that need C/C++)                      #
###############################################################################
apt-get install -y build-essential gcc g++ clang git make cmake

###############################################################################
# 3.  CLONE NEBULA (core branch only)                                          #
###############################################################################
git clone https://github.com/CyberDataLab/nebula.git -b physical-deployment
cd nebula

###############################################################################
# 4.  PYTHON 3.11 VIA “uv” + VIRTUAL-ENV + CORE DEPENDENCIES                   #
###############################################################################
# ---- Install uv -------------------------------------------------------------
curl -fsSL https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"            # make sure uv is on PATH

# Register Python 3.11 as the system default alternatives
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2
update-alternatives --install /usr/bin/python  python  /usr/bin/python3      1

# ---- Create venv and install deps ------------------------------------------
uv python install 3.11.7        # download interpreter if not present
uv python pin 3.11.7            # set default version for uv
uv sync --group core            # install Nebula “core” dependencies
source .venv/bin/activate       # activate the virtual environment

###############################################################################
# 5.  START FASTAPI ***LOCALLY ONLY*** (127.0.0.1:8000)                        #
###############################################################################
# Change “api:app” if your module or variable is named differently.
uvicorn api:app --host 127.0.0.1 --port 8000 --log-level info --workers 1 &

###############################################################################
# 6.  PUBLISH THAT LOCAL PORT TO THE TAILNET **ONLY**                          #
###############################################################################
tailscale serve tcp 8000 &      # tailnet peers reach http://<ts-ip>:8000

###############################################################################
# 7.  DONE – show connection details                                           #
###############################################################################
echo "----------------------------------------------------------"
echo "✅ Nebula API is up."
echo "   Local address : http://127.0.0.1:8000"
echo "   Tailnet address: http://$(tailscale ip -4):8000"
echo "----------------------------------------------------------"
