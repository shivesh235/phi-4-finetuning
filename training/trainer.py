import os
import json
import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import logging

from data.data_module import VQARADDataModule
from models.phi4_vqa_model import Phi4VQAModel
from training.callbacks import ValidationResultsCallback, ModelSizeCallback
from utils.metrics import calculate_vqa_metrics, calculate_binary_metrics

logger = logging.getLogger(__name__)

class VQATrainer:
    def __init__(self, config, checkpoint_path=None, debug_mode=False):
        """
        Initialize the trainer with configuration
        
        Args:
            config: Training configuration
            checkpoint_path: Path to checkpoint for resuming training
            debug_mode: Whether to run in debug mode
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.debug_mode = debug_mode
        
        # Set up output directories
        os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
        os.makedirs(config["output"]["log_dir"], exist_ok=True)
        os.makedirs(config["output"]["prediction_dir"], exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(config["output"]["log_dir"], "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Initialized trainer with config saved to {config_path}")
    
    def setup_model(self):
        """Set up the model for training"""
        logger.info("Setting up model")
        
        # Initialize model
        self.model = Phi4VQAModel(
            model_name=self.config["model"]["name"],
            learning_rate=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
            warmup_steps=self.config["training"]["warmup_steps"],
            lora_config=self.config["model"].get("lora_config", None),
            gradient_checkpointing=self.config["model"].get("gradient_checkpointing", False)
        )
        
        # Initialize data module
        debug_mode = self.debug_mode or self.config["data"].get("debug_mode", False)
        self.data_module = VQARADDataModule(
            processor=self.model.processor,
            train_batch_size=self.config["data"]["train_batch_size"],
            eval_batch_size=self.config["data"]["eval_batch_size"],
            num_workers=self.config["data"]["num_workers"],
            max_length=self.config["data"]["max_length"],
            debug_mode=debug_mode
        )
        
        logger.info("Model and data module setup complete")
    
    def setup_trainer(self):
        """Set up PyTorch Lightning trainer"""
        logger.info("Setting up trainer")
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=os.path.join(self.config["output"]["checkpoint_dir"], "checkpoints"),
                filename="{epoch}-{val_loss:.4f}",
                monitor="val_loss",
                mode="min",
                save_top_k=3,
                save_last=True,
            ),
            EarlyStopping(
                monitor="val_loss",
                patience=self.config["training"]["early_stopping_patience"],
                mode="min",
            ),
            LearningRateMonitor(logging_interval="step"),
            ValidationResultsCallback(num_samples=5),
            ModelSizeCallback()
        ]
        
        # Setup logger
        if self.config["logging"]["use_wandb"]:
            wandb_logger = WandbLogger(
                project=self.config["logging"]["wandb_project"],
                name=self.config["logging"]["wandb_run_name"],
                save_dir=self.config["output"]["log_dir"],
            )
            logger.info(f"WandB logging enabled: {self.config['logging']['wandb_project']}")
        else:
            wandb_logger = None
            logger.info("WandB logging disabled")
        
        # Setup multi-GPU strategy
        if torch.cuda.device_count() > 1:
            strategy = DDPStrategy(find_unused_parameters=False)
            logger.info(f"Using DDP strategy with {torch.cuda.device_count()} GPUs")
        else:
            strategy = None
            logger.info("Using single GPU strategy")
        
        # Initialize PyTorch Lightning trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config["trainer"]["max_epochs"],
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=self.config["trainer"]["gpus"] if torch.cuda.is_available() else None,
            strategy=strategy,
            precision=self.config["trainer"]["precision"] if torch.cuda.is_available() else "32",
            gradient_clip_val=self.config["training"]["gradient_clip_val"],
            accumulate_grad_batches=self.config["training"]["gradient_accumulation_steps"],
            val_check_interval=self.config["training"]["val_check_interval"],
            logger=wandb_logger,
            callbacks=callbacks,
            log_every_n_steps=self.config["logging"]["log_every_n_steps"],
            deterministic=True,
        )
        
        logger.info("Trainer setup complete")
    
    def train(self):
        """Run the training process"""
        logger.info("Starting training")
        
        self.setup_model()
        self.setup_trainer()
        
        # Start training
        if self.checkpoint_path:
            logger.info(f"Resuming from checkpoint: {self.checkpoint_path}")
            self.trainer.fit(self.model, datamodule=self.data_module, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model, datamodule=self.data_module)
        
        # Test the model after training
        logger.info("Starting model evaluation")
        test_results = self.trainer.test(self.model, datamodule=self.data_module)
        
        # Save evaluation results
        results_path = os.path.join(self.config["output"]["prediction_dir"], "test_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Training completed. Test results saved to {results_path}")
        
        return test_results
    
    def evaluate(self, checkpoint_path):
        """
        Evaluate a trained model
        
        Args:
            checkpoint_path: Path to the model checkpoint
        """
        logger.info(f"Evaluating model from checkpoint: {checkpoint_path}")
        
        self.setup_model()
        self.setup_trainer()
        
        # Load model from checkpoint
        logger.info("Loading model from checkpoint")
        self.model = Phi4VQAModel.load_from_checkpoint(checkpoint_path)
        
        # Run evaluation
        test_results = self.trainer.test(self.model, datamodule=self.data_module)
        
        # Process detailed predictions
        logger.info("Processing detailed predictions")
        predictions_df = self._process_predictions()
        
        # Save predictions
        predictions_path = os.path.join(self.config["output"]["prediction_dir"], "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)
        
        # Save evaluation results
        results_path = os.path.join(self.config["output"]["prediction_dir"], "evaluation_results.json")
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_path}")
        
        return test_results, predictions_df
    
    def _process_predictions(self):
        """Process and analyze model predictions"""
        # Combine predictions, ground truth and answer types
        data = {
            "prediction": self.model.predictions,
            "ground_truth": self.model.ground_truth,
            "answer_type": self.model.answer_types,
            "correct": [p.strip() == r.strip() for p, r in zip(self.model.predictions, self.model.ground_truth)]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate metrics for closed questions
        close_df = df[df["answer_type"] == "CLOSE"]
        if not close_df.empty:
            close_metrics = calculate_binary_metrics(close_df["prediction"].tolist(), close_df["ground_truth"].tolist())
            logger.info(f"Closed-ended questions metrics: {close_metrics}")
        
        # Calculate overall metrics
        metrics = calculate_vqa_metrics(df["prediction"].tolist(), df["ground_truth"].tolist(), df["answer_type"].tolist())
        logger.info(f"Overall metrics: {metrics}")
        
        return df