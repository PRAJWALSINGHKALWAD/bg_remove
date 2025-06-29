# GitHub Actions Background Removal - Usage Instructions

## Quick Start Guide

### 1. Add Images for Processing

Place your images in one of these ways:

**Option A: Upload to `input/` folder (Recommended)**
```
input/
├── photo1.jpg
├── product-image.png
└── portrait.jpeg
```

**Option B: Upload to root directory**
- Just drag and drop images to the main folder
- The workflow will find them automatically

### 2. Run the Workflow

1. Go to **Actions** tab in GitHub
2. Click **"Professional Background Removal"**
3. Click **"Run workflow"** 
4. Select options:
   - **Quality Level**: `ultra` (recommended) or `commercial` (best quality)
   - **Batch Size**: `10` (default, good for most cases)

### 3. Download Results

1. Wait for workflow to complete (green checkmark)
2. Scroll down to **Artifacts** section
3. Download **"professional-background-removal-complete"**
4. Extract the ZIP file to get all processed images

## Quality Level Guide

| Level | Best For | Processing Time | Quality |
|-------|----------|----------------|---------|
| `fast` | Quick previews | ~30s per image | Good |
| `high` | General use | ~60s per image | Very Good |
| `ultra` | Professional work | ~90s per image | Excellent |
| `commercial` | Remove.bg quality | ~120s per image | Perfect |

## Example Processing Times

- **10 images (ultra quality)**: ~15 minutes
- **50 images (high quality)**: ~50 minutes  
- **100 images (fast quality)**: ~50 minutes

## Tips for Best Results

1. **Image Quality**: Use high-resolution images (1000px+ width)
2. **Subject Contrast**: Ensure good contrast between subject and background
3. **Batch Size**: For large batches (50+ images), use batch size of 5-8
4. **Quality Choice**: Start with `ultra`, upgrade to `commercial` if needed

## Troubleshooting

- **Workflow fails**: Try smaller batch size or lower quality
- **Poor results**: Use higher quality level or check source image
- **Timeout**: Reduce batch size to 5 images per batch

---

Ready to process? Upload your images and run the workflow! 🚀
