# Changelog

All notable changes to Video Buddy are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/) once 1.0 ships.

## [Unreleased]

### Managed ComfyUI

- CPU mode works end to end: on a machine without a usable GPU, `comfy start` launches ComfyUI with `--cpu`, so a CPU-only install no longer stops at startup. `COMFY_CPU` overrides the detection.

### Install

- The automatic install pre-flight checks that git is installed and stops with the install command for your system when it is not.

### Documentation

- A "Before you start" list with the real prerequisites: Python 3.10+ with `venv`, git, internet, about 100 GB of free disk, and an NVIDIA driver for GPU speed (no CUDA toolkit). Without a GPU, Buddy runs in CPU mode, which is very slow.
