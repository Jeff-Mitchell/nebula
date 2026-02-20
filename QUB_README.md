# QUB Changes

This repository reflects **QUB (Queen’s University Belfast) fork-specific development** on top of the original upstream NEBULA project.

## Summary
- **Improvements and expansion of attack logic for experiments**
  - Strengthened attack behaviour implementation for intermittent or intercepted attacks.

- **Better security evaluation and reporting of metrics**
  - Added clearer evaluation of attack success rates (including targeted label-flipping and backdoor attacks).

- **Automation of experiments and operations**
  - Added scripting support for submitting, queuing, and deleting scenarios without manual backend interaction.

- **Safer container/runtime behaviour**
  - Improved GPU mapping logic for node containers, with safer defaults and reduced misconfiguration risk.

- **Reliability and bug fixes throughout**
  - Fixed data/shape issues such as incorrect image dimension handling in some image datasets.
  - Addressed multiple issues in topology, aggregation, and evaluation pipelines.

- **Overall fork direction**
  - This branch prioritises QUB research needs towards:
  - More robust adversarial experimentation.
  - Clearer security evaluation.
  - More practical experiment operations.

# Setting up NEBULA

Based on the official NEBULA installation guide: https://docs.nebula-dfl.com/installation/

## Prerequisites
- Linux (Ubuntu 20.04 LTS recommended) or macOS 10.15+.
- At least 8 GB RAM (32 GB recommended for virtualized devices).
- At least 20 GB free disk space (plus extra for datasets/models/results).
- Docker Engine >= 24.0.4 (24.0.7 recommended).
- Docker Compose >= 2.19.0 (2.19.1 recommended).

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd nebula
   ```
2. Install dependencies and set up containers:
   ```bash
   make install
   ```
3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```
   If needed, open the project shell with:
   ```bash
   make shell
   ```

## Optional (NVIDIA GPU nodes)
- NVIDIA Driver >= 525.60.13.
- CUDA 12.1 (verify with `nvidia-smi`).
- NVIDIA Container Toolkit for Docker GPU access.

## Usage
- Verify the installation:
  ```bash
  python app/main.py --version
  ```

- Run NEBULA:
  ```bash
  python app/main.py
  ```

- Show available options:
  ```bash
  python app/main.py --help
  ```

- Frontend default: `http://127.0.0.1:6060` (random port if 6060 is unavailable).

- Set custom ports:
  ```bash
  python app/main.py --webport [PORT]
  python app/main.py --statsport [PORT]
  ```

- Default frontend credentials:
  > **Username:** `admin`
  > **Password:** `admin`

- Stop NEBULA:
  ```bash
  python app/main.py --stop
  ```
