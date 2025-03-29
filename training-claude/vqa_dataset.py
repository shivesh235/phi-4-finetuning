import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class MedicalVQADataset(Dataset):
    def __init__(self, data_path, image_dir, tokenizer, image_processor):
        """
        Custom Dataset for Medical Visual Question Answering
        
        Args:
            data_path (str): Path to JSON file containing VQA annotations
            image_dir (str): Directory containing medical images
            tokenizer: Hugging Face tokenizer
            image_processor: Image preprocessing function
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # Add special tokens for instruction tuning
        special_tokens = ["[QUESTION]", "[IMAGE]", "[ANSWER]"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_path = f"{self.image_dir}/{item['image_filename']}"
        image = Image.open(image_path).convert('RGB')
        processed_image = self.image_processor(image)
        
        # Prepare input text
        input_text = f"[QUESTION] {item['question']} [IMAGE]"
        
        # Tokenize input
        input_encodings = self.tokenizer(
            input_text, 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Tokenize answer
        answer_encodings = self.tokenizer(
            f"[ANSWER] {item['answer']}", 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encodings['input_ids'].flatten(),
            'attention_mask': input_encodings['attention_mask'].flatten(),
            'image': processed_image,
            'labels': answer_encodings['input_ids'].flatten()
        }

def prepare_medical_vqa_dataset(data_path, image_dir, tokenizer, image_processor, test_size=0.2):
    """
    Prepare train and validation splits for medical VQA dataset
    
    Args:
        data_path (str): Path to JSON file with VQA data
        image_dir (str): Directory containing medical images
        tokenizer: Hugging Face tokenizer
        image_processor: Image preprocessing function
        test_size (float): Proportion of data to use for validation
    
    Returns:
        Tuple of train and validation datasets
    """
    from sklearn.model_selection import train_test_split
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    
    # Temporarily save splits
    with open('train_split.json', 'w') as f:
        json.dump(train_data, f)
    with open('val_split.json', 'w') as f:
        json.dump(val_data, f)
    
    train_dataset = MedicalVQADataset(
        'train_split.json', 
        image_dir, 
        tokenizer, 
        image_processor
    )
    
    val_dataset = MedicalVQADataset(
        'val_split.json', 
        image_dir, 
        tokenizer, 
        image_processor
    )
    
    return train_dataset, val_dataset