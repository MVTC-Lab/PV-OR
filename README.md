# PV-Orchestrator: Reprogramming Frozen Foundation Models for Solar Forecasting via Neural Prompting and Mixture-of-Experts

[![license](https://img.shields.io/badge/License-MIT-green)](#license)
## 🔥 Overview
This repository provides the reference implementation for **PV-Orchestrator**

**Key ideas**
- **Frozen LLM backbone** with lightweight adaptation (<P-Tuning / Prompt / Reprogramming>).
- **MoE** for condition-aware routing.
- **Environmental Transformer Branch** to improve robustness under distribution shifts.

> Model architecture figure: see
> ![`figures/model_framwork.png`](figures/model_framwork.png)

---

## ✨ Features
- Reproducible training & evaluation scripts
- Multi-dataset support
- Parameter-efficient adaptation (no full fine-tuning)
- Logging with <TensorBoard/W&B> and checkpointing
---

## Environment Setup (uv / conda)
Option A: uv (recommended)

uv manages dependencies via pyproject.toml and pins exact versions in uv.lock. uv sync will create a project virtual environment (typically .venv/) and install dependencies
```bash
# 1) Install uv (choose ONE)
pip install -U uv
# or (macOS/Linux) curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) Create/sync the environment from uv.lock
uv sync

# 3) Run scripts inside the managed environment
uv run python train.py --config configs/<dataset>.yaml
uv run python eval.py  --config configs/<dataset>.yaml --ckpt <path>
```
Notes

By default, uv sync installs the local project in editable mode, so your code changes take effect immediately. 
Astral 文档

If you want a requirements.txt for compatibility (not recommended if you already use uv.lock), you can export from the lockfile:
```bash
uv export --format requirements.txt --output-file requirements.txt
```

Option B: conda (compatible / traditional)

Use conda to create a clean Python environment, then install dependencies from pyproject.toml.
```bash
# 1) Create environment (match the Python version in pyproject.toml)
conda create -n <env_name> python=3.13 -y
conda activate <env_name>

# 2) Install dependencies
pip install -U pip
pip install -e .
```






