#!/bin/bash

# Setup script for MMDetection following best practices
echo "Installing MMDetection following OpenMMLab best practices..."

# Set PEP 517 environment variable to avoid legacy build issues
# export PIP_USE_PEP517=true

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first or use the pip fallback option."
    echo "Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if we're in a virtual environment or uv project
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment detected: $VIRTUAL_ENV"
elif [[ -f "uv.lock" ]]; then
    echo "UV project detected, using uv run..."
else
    echo "Warning: No virtual environment or uv project detected."
fi

echo "=== Step 0: Install MMEngine and MMCV using MIM ==="

# Install OpenMIM first
# echo "Installing OpenMIM..."
# uv add openmim

# Install MMEngine using MIM
echo "Installing MMEngine..."
uv run mim install mmengine

# Install MMCV lite version (without CUDA ops) using MIM
echo "Installing MMCV"
uv run mim install "mmcv>=2.0.0rc4, <2.2.0"

echo "=== Step 1: Install MMDetection from source ==="
uv run mim install "mmdet"
# # Create a directory for MMDetection if it doesn't exist
# if [ ! -d "mmdetection" ]; then
#     echo "Cloning MMDetection repository..."
#     git clone https://github.com/open-mmlab/mmdetection.git
# else
#     echo "MMDetection directory already exists, updating..."
#     cd mmdetection
#     git pull origin main
#     cd ..
# fi

# # Install MMDetection in editable mode
# echo "Installing MMDetection in editable mode..."
# cd mmdetection
# uv run pip install -v -e .
# cd ..

echo ""
echo "=== MMDetection setup complete! ==="
echo ""
echo "Installation summary:"
echo "- OpenMIM: installed"
echo "- MMEngine: installed via MIM"
echo "- MMCV: lite version installed (without CUDA ops)"
echo "- MMDetection: installed from source in editable mode"
echo ""
echo "To verify the installation, run:"
echo "uv run python -c 'import mmdet; print(mmdet.__version__)'"
echo "uv run python -c 'import mmengine; print(mmengine.__version__)'"
echo "uv run python -c 'import mmcv; print(mmcv.__version__)'"
echo ""
echo "If you encounter issues, check the MMDetection documentation:"
echo "https://mmdetection.readthedocs.io/en/latest/get_started.html"
