# X-Ray Report Generation Pipeline
A unified pipeline for automated chest X-ray analysis combining zero-shot pathology classification, global impression generation, and local findings generation using deep learning models.

## Features

- 🔍 **Zero-shot Pathology Classification**: Detects multiple pathologies without task-specific training using CLIP  
- 💬 **Global Impression Generation**: Creates overall radiology impressions from X-ray images  
- 🔎 **Local Findings Generation**: Generates detailed, region-specific findings  
- 📊 **Batch Processing**: Efficiently processes multiple images  
- 💾 **Structured Output**: Results saved in CSV format with predictions and generated text  

## Model Architecture
This pipeline uses three specialized models:

- **CheXzero (CLIP-based)**: Pre-trained vision-language model for zero-shot classification  
- **Global Model**: Encoder-decoder architecture for generating overall impressions  
- **Local Model**: Fine-grained model for detailed regional findings  

## Installation

### Prerequisites
Python 3.8+  
CUDA-capable GPU (recommended)  
8GB+ GPU memory  

### Setup

#### Clone the repository
```bash
git clone https://github.com/nameer451/xray-report-generation.git
cd xray-report-generation
```

## Install dependencies
```bash
# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install OpenAI CLIP
pip install ftfy regex tqdm


pip install git+https://github.com/openai/CLIP.git

# Install other requirements
pip install h5py pandas numpy Pillow matplotlib opencv-python
pip install transformers sentencepiece
pip install albumentations huggingface-hub
```

## Download model checkpoints

Download the model weights and place them in the `checkpoints/` folder:

- CheXzero weights: `best_64_0.0001_original_35000_0.864.pt`
- Global decoder weights: `global_decoder_epoch5.pt`
- Local model weights: `stage2_final.pt`

[Download Link: https://drive.google.com/drive/folders/136zvREdD0wFmVTZnkjvqEagsgipVZG8N?usp=drive_link]

Your directory structure should look like:

```bash
xray-report-generation/
├── run_pipeline.py
├── checkpoints/
│   ├── best_64_0.0001_original_35000_0.864.pt
│   ├── global_decoder_epoch5.pt
│   └── stage2_final.pt
├── data/
│   └── xray_images/          # Put your X-ray images here
├── global/
│   ├── global_model.py
│   └── ...
└── local/
    ├── local_model.py
    └── ...
```

# Usage

## Quick Start

Place your X-ray images in `./data/xray_images/`

Supported formats: `.jpg`, `.jpeg`, `.png`, `.dcm`

### Run the pipeline
```bash
python run_pipeline.py
```
View results in ./results/

results.csv: All predictions and generated text

Individual image reports with pathology probabilities
