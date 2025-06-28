from PIL import Image, UnidentifiedImageError
from rembg import remove
import os
import sys

# Usage: python bg_remove.py <input_image> [output_dir]
if len(sys.argv) < 2:
    print("Usage: python bg_remove.py <input_image> [output_dir]")
    sys.exit(1)

input_path = sys.argv[1]
output_dir = sys.argv[2] if len(sys.argv) > 2 else ''

# Get file extension and base name
base = os.path.splitext(os.path.basename(input_path))[0]
ext = os.path.splitext(input_path)[1].lower()

# Set output directory
if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{base}_nobg{ext}")
else:
    output_path = os.path.join(os.path.dirname(input_path), f"{base}_nobg{ext}")

try:
    img = Image.open(input_path)
except UnidentifiedImageError:
    print(f"Warning: Cannot identify image file '{input_path}'. Skipping.")
    sys.exit(0)

result = remove(img)

# Convert to RGB if saving as JPEG
if ext in ['.jpg', '.jpeg']:
    result = result.convert('RGB')

result.save(output_path)
print(f"Saved: {output_path}")
