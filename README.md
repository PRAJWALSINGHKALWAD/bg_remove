# Background Removal Script

This project removes backgrounds from images using AI-powered technology. All dependencies are managed locally in a virtual environment to keep your system clean.

## Project Structure

```
bg_remove/
├── bg-remove.py          # Main script for background removal
├── requirements.txt      # Project dependencies
├── venv/                # Virtual environment (all packages installed here)
├── run.bat              # Windows batch script to run the project
├── run.ps1              # PowerShell script to run the project
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```

## Features

- Remove backgrounds from images using AI
- Support for multiple input formats (JPG, JPEG, PNG, etc.)
- **Always outputs PNG format to preserve transparency**
- Transparent background (not black or white)
- Optional output directory specification
- All packages isolated in project-specific virtual environment

## Usage

### Method 1: Using the batch script (Recommended)
```bash
run.bat "image.jpg"
run.bat "image.jpg" "output_folder"
```

### Method 2: Using PowerShell script
```powershell
.\run.ps1 "image.jpg"
.\run.ps1 "image.jpg" "output_folder"
```

### Method 3: Manual activation
```bash
# Activate virtual environment
venv\Scripts\activate

# Run the script
python bg-remove.py "image.jpg"
python bg-remove.py "image.jpg" "output_folder"

# Deactivate when done
deactivate
```

## Installation

The virtual environment and all dependencies are already installed. If you need to reinstall:

```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- **Pillow**: Image processing library
- **rembg**: AI-powered background removal
- **onnxruntime**: Required for AI model execution
- Additional dependencies: numpy, opencv-python-headless, scikit-image, etc.

All packages are installed in the local `venv` folder, not in your system Python installation.

## Examples

```bash
# Remove background from a single image
run.bat "photo.jpg"
# Output: photo_nobg.png (transparent background)

# Remove background and save to specific directory
run.bat "photo.jpg" "cleaned_images"
# Output: cleaned_images/photo_nobg.png
```

## Notes

- The first run may take longer as the AI model downloads
- **Output files are always PNG format with transparent backgrounds**
- Input can be any image format (JPG, PNG, etc.)
- Output files are named with "_nobg" suffix
- The virtual environment keeps all packages isolated from your system
