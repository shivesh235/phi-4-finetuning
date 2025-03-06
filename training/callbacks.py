import os
import torch
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks import Callback
import logging

logger = logging.getLogger(__name__)

class ValidationResultsCallback(Callback):
    """
    Callback to log validation results and sample predictions
    """
    def __init__(self, num_samples=5):
        super().__init__()
        self.num_samples = num_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.predictions or not pl_module.ground_truth:
            return
        
        # Select a random subset of predictions and ground truths for logging
        indices = np.random.choice(
            len(pl_module.predictions), 
            min(self.num_samples, len(pl_module.predictions)), 
            replace=False
        )
        
        # Create a DataFrame for logging
        samples = []
        for i in indices:
            samples.append({
                "Prediction": pl_module.predictions[i],
                "Ground Truth": pl_module.ground_truth[i],
                "Answer Type": pl_module.answer_types[i],
                "Correct": pl_module.predictions[i].strip() == pl_module.ground_truth[i].strip()
            })
        
        # Log samples to console
        logger.info(f"Validation epoch {trainer.current_epoch} sample results:")
        for i, sample in enumerate(samples):
            logger.info(f"Sample {i+1}:")
            logger.info(f"  Question Type: {sample['Answer Type']}")
            logger.info(f"  Prediction: {sample['Prediction']}")
            logger.info(f"  Ground Truth: {sample['Ground Truth']}")
            logger.info(f"  Correct: {sample['Correct']}")
        
        # Log to WandB if available
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            try:
                trainer.logger.experiment.log({
                    "validation_samples": pd.DataFrame(samples),
                    "epoch": trainer.current_epoch
                })
            except:
                pass


class ModelSizeCallback(Callback):
    """
    Callback to monitor and log model's memory usage
    """
    def on_train_start(self, trainer, pl_module):
        # Log model size at the beginning of training
        model_size_mb = self._get_model_size_mb(pl_module)
        logger.info(f"Model size: {model_size_mb:.2f} MB")
        
        if trainer.logger and hasattr(trainer.logger, 'experiment'):
            try:
                trainer.logger.experiment.log({"model_size_mb": model_size_mb})
            except:
                pass
    
    def _get_model_size_mb(self, model):
        """Calculate model size in MB"""
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        model_size = sum([p.numel() * p.element_size() for p in model_parameters])
        return model_size / (1024 * 1024)  # Convert to MB