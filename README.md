# Phi-4 Medical VQA Fine-tuning

This repository contains code for fine-tuning Microsoft's Phi-4 Vision model on medical visual question answering (VQA) datasets. The implementation supports efficient training techniques including parameter-efficient fine-tuning (PEFT) via LoRA, mixed-precision training, gradient accumulation, and multi-GPU training.

## Overview

The project enables fine-tuning the Phi-4 multimodal model to answer questions about medical images. It's designed to be flexible, efficient, and easy to use, with focus on high performance and reproducibility.

## Features

- **Parameter-Efficient Fine-Tuning**: Uses LoRA to fine-tune selected parameters
- **Multi-GPU Support**: Distributed training with PyTorch Lightning
- **Mixed Precision**: BF16/FP16 training for faster performance
- **Comprehensive Metrics**: Detailed evaluation for both closed and open-ended questions
- **Experiment Tracking**: WandB integration for experiment monitoring
- **Optimized Data Loading**: Efficient data processing pipeline

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- 32GB+ RAM recommended

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/phi4-medical-vqa.git
cd phi4-medical-vqa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

1. Configure training parameters in `config/training_config.yaml`

2. Run training:
```bash
python main.py
```

For debugging or quick tests:
```bash
python main.py --debug
```

To resume from a checkpoint:
```bash
python main.py --checkpoint path/to/checkpoint.ckpt
```

### Multi-GPU Training

The code automatically uses all available GPUs with DDP strategy. You can specify the number of GPUs in the config file or override it:

```bash
python main.py --config config/training_config.yaml
```

## Project Structure

```
project_root/
├── config/               # Configuration files
├── data/                 # Data loading and processing
├── models/               # Model definitions
├── training/             # Training utilities and callbacks
├── utils/                # Helper functions and metrics
├── main.py               # Entry point
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Configuration

The main configuration file is `config/training_config.yaml`. Key configuration sections include:

- **Model Configuration**: Model architecture and optimization settings
- **Training Parameters**: Learning rates, weight decay, etc.
- **Data Processing**: Batch sizes, sequence lengths, etc.
- **Trainer Settings**: Epochs, precision, etc.
- **Logging**: WandB configuration and logging frequency
- **Output**: Paths for checkpoints, logs, and predictions

## Metrics and Evaluation

The implementation calculates several metrics for evaluation:
- Exact match accuracy
- BLEU score
- ROUGE scores
- Specialized metrics for closed-ended (yes/no) questions

## Extending

### Adding New Datasets

To add a new dataset:
1. Create a new dataset class in `data/`
2. Implement the required methods: `__len__`, `__getitem__`
3. Update the data module to use your new dataset

### Using Different Models

To use a different model:
1. Update the model name in the configuration file
2. Ensure the processor and model architecture are compatible
3. Modify the model class if necessary to accommodate differences

## License

This project is released under the MIT License.

## Acknowledgements

- Microsoft for the Phi-4 Vision model
- Hugging Face for transformers and datasets libraries
- PyTorch team for PyTorch and PyTorch Lightning