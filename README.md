# MBTI Personality Type Classification using Transformers

This project uses transformer models (RoBERTa and DistilRoBERTa) to classify MBTI personality types based on text input.

## Project Structure
```
├── Train.py              # Main training script
├── util.py               # Utility script for NLTK setup
├── requirements.txt      # Python package dependencies
├── README.md             # This file
├── mbti_cleaned.csv      # Your dataset file
├── .ipynb file           # The preprocessing and EDA files
```

## Installation

1. **Clone the repository** (if applicable) or download the files

2. **Install Python packages:**
```bash
pip install -r requirements.txt
```

3. **Install NLTK dependencies:**
```bash
python util.py
```

## Dataset
Place your `mbti_cleaned.csv` file in the root directory. The CSV should contain:
- `type`: MBTI personality type (e.g., INTJ, ENFP, etc.)
- `cleaned_posts`: Preprocessed text data

## Usage

Run the main training script:
```bash
python Train.py
```

This will:
1. Train both RoBERTa and DistilRoBERTa models
2. Perform grid search validation
3. Train final models with best hyperparameters
4. Save best checkpoints
5. Evaluate on test set
6. Generate detailed performance reports

## Model Configuration

**Best Hyperparameters Found:**
- Optimizer: AdamW
- Learning Rate: 5e-5
- Warmup Steps: 100
- Batch Size: 64
- Epochs: 10

## Requirements

- Python 3.9+
- NVIDIA A100 40G GPU