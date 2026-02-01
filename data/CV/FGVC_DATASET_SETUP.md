# FGVC Aircraft Dataset Setup Guide

## Step 1: Download the Dataset

### Official Source (Recommended)
1. Go to https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
2. Download the dataset using the links provided on the page
3. Extract to `data/CV/aircraft/`

### Quick Download Commands
```bash
# Create directory
mkdir -p data/CV/aircraft
cd data/CV/aircraft

# Download the main dataset (contains images and annotations)
# Follow the links on https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
# Download: fgvc-aircraft-2013b.tar.gz or fgvc-aircraft-2013b.zip

# Extract (if tar.gz)
tar -xzf fgvc-aircraft-2013b.tar.gz

# Or extract (if zip)
unzip fgvc-aircraft-2013b.zip
```

### Alternative: Kaggle Mirror (if official source unavailable)
```bash
pip install kaggle
kaggle datasets download -d ducha/fgvc-aircraft-2013-dataset -p data/CV/
unzip data/CV/fgvc-aircraft-2013-dataset.zip
```

## Expected Directory Structure

After extraction, your structure should be:
```
data/CV/aircraft/
├── data/
│   ├── images/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── variants.txt
│   ├── familiesHierarchy.txt
│   └── familiesHierarchy_v2.txt
├── splits/
│   ├── images_family_test.txt
│   ├── images_family_train.txt
│   └── images_family_val.txt
└── (other split files)
```

## Dataset Information

- **Total Images**: ~10,000
- **Aircraft Families**: 10
- **Aircraft Models**: 100
- **Image Format**: JPG
- **Annotations**: Text files with image-to-family and image-to-model mappings

## Integration in Your Project

See `src/CNN/dataset_loader.py` for loading functions that integrate with your training pipeline.
