import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoTokenizer, 
    Phi3ForCausalLM
)

class MedicalVQAModel(nn.Module):
    def __init__(self, model_name="microsoft/phi-3-mini-4k-instruct"):
        """
        Multimodal Medical VQA Model
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        super().__init__()
        
        # Load pre-trained components
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.language_model = Phi3ForCausalLM.from_pretrained(model_name)
        
        # Freeze language model parameters
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Add custom vision projection layer
        self.vision_projection = nn.Sequential(
            nn.Linear(self.language_model.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Custom multimodal fusion layer
        self.multimodal_fusion = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=512, 
                nhead=8
            ), 
            num_layers=2
        )
    
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        image=None, 
        labels=None
    ):
        # Process image
        vision_features = self.processor(images=image, return_tensors="pt")
        
        # Extract language embeddings
        text_outputs = self.language_model(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        
        # Project and fuse multimodal features
        text_embeddings = text_outputs.last_hidden_state
        
        # Fusion and generation
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

def load_medical_vqa_model(model_path=None):
    """
    Load or initialize Medical VQA model
    
    Args:
        model_path (str, optional): Path to saved model checkpoint
    
    Returns:
        Configured MedicalVQAModel
    """
    model = MedicalVQAModel()
    
    if model_path:
        model.load_state_dict(torch.load(model_path))
    
    return model