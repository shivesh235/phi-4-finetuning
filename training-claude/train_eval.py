import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torchmetrics import Accuracy
import wandb

def train_medical_vqa_model(
    model, 
    train_dataset, 
    val_dataset, 
    learning_rate=2e-5, 
    batch_size=4, 
    num_epochs=5
):
    """
    Train Medical VQA Model
    
    Args:
        model (MedicalVQAModel): Model to train
        train_dataset (Dataset): Training dataset
        val_dataset (Dataset): Validation dataset
        learning_rate (float): Initial learning rate
        batch_size (int): Training batch size
        num_epochs (int): Number of training epochs
    """
    # Prepare data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=len(train_loader) * num_epochs
    )
    
    # Metrics
    accuracy = Accuracy(task='multiclass', num_classes=model.tokenizer.vocab_size)
    
    # Optional: Initialize wandb for experiment tracking
    wandb.init(project="medical-vqa", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs
    })
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image=batch['image'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_accuracy = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    image=batch['image'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                # Calculate accuracy
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                val_accuracy += accuracy(preds, batch['labels'])
        
        # Log metrics
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        
        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": avg_val_accuracy
        })
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {avg_val_accuracy:.4f}")
    
    # Save model
    torch.save(model.state_dict(), "medical_vqa_model.pth")
    wandb.finish()

def evaluate_medical_vqa_model(model, test_dataset):
    """
    Evaluate trained Medical VQA model
    
    Args:
        model (MedicalVQAModel): Trained model
        test_dataset (Dataset): Test dataset
    
    Returns:
        Dict with evaluation metrics
    """
    test_loader = DataLoader(test_dataset, batch_size=4)
    
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                image=batch['image'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            correct_predictions += (preds == batch['labels']).sum().item()
            total_predictions += batch['labels'].numel()
    
    metrics = {
        "test_loss": total_loss / len(test_loader),
        "accuracy": correct_predictions / total_predictions,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions
    }
    
    print("Test Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    return metrics

# Main execution script
def main():
    # Load model and processor
    model = load_medical_vqa_model()
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_medical_vqa_dataset(
        data_path='medical_vqa_data.json',
        image_dir='medical_images/',
        tokenizer=model.tokenizer,
        image_processor=model.processor
    )
    
    # Train the model
    train_medical_vqa_model(
        model, 
        train_dataset, 
        val_dataset
    )
    
    # Evaluate the model
    test_dataset = MedicalVQADataset(
        'test_split.json', 
        'medical_images/', 
        model.tokenizer, 
        model.processor
    )
    evaluate_medical_vqa_model(model, test_dataset)

if __name__ == "__main__":
    main()