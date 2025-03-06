import os
import torch
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from datasets import load_dataset
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class VQARADDataset(Dataset):
    def __init__(self, dataset_split, processor, max_length=2048, debug_mode=False):
        self.dataset = dataset_split
        self.processor = processor
        self.max_length = max_length
        
        # If in debug mode, use only a small subset of data
        if debug_mode:
            logger.info(f"Debug mode enabled, using only 10 samples from {len(self.dataset)} total")
            self.dataset = Subset(self.dataset, range(min(10, len(self.dataset))))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load image
        image = item['image'].convert('RGB')

        # Construct prompt
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a medical assistant who analyses the image and answers the question in very few words."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": item['question']},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Tokenize prompt and process image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True
        )
        
        # Determine answer type
        if item['answer'].lower() in ['yes', 'no']:
            answer_type = 'CLOSE'
        else:
            answer_type = 'OPEN'

        # Tokenize the answer
        target = self.processor.tokenizer(
            item['answer'],
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
        ).input_ids.squeeze(0)

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0),
            "labels": target,
            "answer_type": answer_type
        }


class VQARADDataModule(pl.LightningDataModule):
    def __init__(
        self,
        processor,
        train_batch_size: int = 8,
        eval_batch_size: int = 8,
        num_workers: int = 4,
        max_length: int = 2048,
        debug_mode: bool = False
    ):
        """
        Args:
            processor: Phi-4 processor for text and image processing
            train_batch_size: Batch size for training
            eval_batch_size: Batch size for validation/testing
            num_workers: Number of workers for data loading
            max_length: Maximum sequence length for tokenization
            debug_mode: If True, use a small subset of data for debugging
        """
        super().__init__()
        self.processor = processor
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.max_length = max_length
        self.debug_mode = debug_mode
        self.save_hyperparameters(ignore=['processor'])

    def setup(self, stage: Optional[str] = None):
        # Load dataset from Hugging Face
        logger.info("Loading VQA-RAD dataset")
        dataset = load_dataset("flaviagiammarino/vqa-rad")
        
        # Create dataset instances
        self.train_dataset = VQARADDataset(
            dataset["train"], 
            self.processor, 
            self.max_length,
            self.debug_mode
        )
        
        self.val_dataset = VQARADDataset(
            dataset["validation"] if "validation" in dataset else dataset["test"], 
            self.processor, 
            self.max_length,
            self.debug_mode
        )
        
        self.test_dataset = VQARADDataset(
            dataset["test"], 
            self.processor, 
            self.max_length,
            self.debug_mode
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )