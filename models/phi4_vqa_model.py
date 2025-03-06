import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model
import evaluate
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Phi4VQAModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "microsoft/Phi-4-vision-32k",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        lora_config: Optional[Dict[str, Any]] = None,
        gradient_checkpointing: bool = True,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Load Phi-4 model and processor
        logger.info(f"Loading model and processor: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Enable gradient checkpointing to save memory
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        
        # Apply LoRA if configured
        if lora_config:
            logger.info(f"Applying LoRA with config: {lora_config}")
            lora_config = LoraConfig(
                r=lora_config.get("r", 16),
                lora_alpha=lora_config.get("lora_alpha", 32),
                target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=lora_config.get("lora_dropout", 0.05),
                bias=lora_config.get("bias", "none"),
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Set up metrics
        self.accuracy_metric = evaluate.load("accuracy")
        self.exact_match_metric = evaluate.load("exact_match")
        
        # For validation and test predictions
        self.ground_truth = []
        self.predictions = []
        self.raw_outputs = []
        self.answer_types = []
    
    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Generate answers for evaluation
        gen_kwargs = {
            "max_new_tokens": 64,
            "num_beams": 3,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": False
        }
        
        # Generation
        generated_ids = self.model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pixel_values=batch["pixel_values"],
            **gen_kwargs
        )
        
        # Decode the generated tokens
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        reference_texts = self.processor.batch_decode(batch["labels"], skip_special_tokens=True)
        
        # Store for metrics calculation
        self.ground_truth.extend(reference_texts)
        self.predictions.extend(generated_texts)
        self.answer_types.extend(batch["answer_type"])
        
        return loss
    
    def on_validation_epoch_end(self):
        # Calculate metrics
        if self.ground_truth and self.predictions:
            # Calculate overall metrics
            accuracy = self.accuracy_metric.compute(
                predictions=self.predictions,
                references=self.ground_truth
            )
            exact_match = self.exact_match_metric.compute(
                predictions=self.predictions,
                references=self.ground_truth
            )
            
            # Calculate metrics by answer type
            close_preds = [p for p, t in zip(self.predictions, self.answer_types) if t == "CLOSE"]
            close_refs = [r for r, t in zip(self.ground_truth, self.answer_types) if t == "CLOSE"]
            open_preds = [p for p, t in zip(self.predictions, self.answer_types) if t == "OPEN"]
            open_refs = [r for r, t in zip(self.ground_truth, self.answer_types) if t == "OPEN"]
            
            # Calculate metrics for closed and open answers if available
            close_accuracy = self.accuracy_metric.compute(
                predictions=close_preds, references=close_refs
            ) if close_preds else {"accuracy": 0.0}
            
            open_accuracy = self.accuracy_metric.compute(
                predictions=open_preds, references=open_refs
            ) if open_preds else {"accuracy": 0.0}
            
            # Log metrics
            self.log("val_accuracy", accuracy["accuracy"], prog_bar=True, sync_dist=True)
            self.log("val_exact_match", exact_match["exact_match"], prog_bar=True, sync_dist=True)
            self.log("val_close_accuracy", close_accuracy["accuracy"], sync_dist=True)
            self.log("val_open_accuracy", open_accuracy["accuracy"], sync_dist=True)
            
            # Clear lists for next validation
            self.ground_truth = []
            self.predictions = []
            self.answer_types = []
    
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
    
    def configure_optimizers(self):
        # Set up optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Set up learning rate scheduler
        total_steps = self.trainer.estimated_stepping_batches
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=total_steps,
        )
        
        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_loss",
            "strict": True,
            "name": "learning_rate",
        }
        
        return [optimizer], [scheduler_config]