import torch
from PIL import Image

class MedicalVQAInference:
    def __init__(self, model_path):
        """
        Inference utility for Medical Visual Question Answering
        
        Args:
            model_path (str): Path to trained model checkpoint
        """
        self.model = load_medical_vqa_model(model_path)
        self.model.eval()
    
    def predict(self, image_path, question):
        """
        Generate answer for a given medical image and question
        
        Args:
            image_path (str): Path to medical image
            question (str): Question about the image
        
        Returns:
            str: Generated answer
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        processed_image = self.model.processor(image, return_tensors="pt")
        
        # Prepare input text
        input_text = f"[QUESTION] {question} [IMAGE]"
        input_encodings = self.model.tokenizer(
            input_text, 
            max_length=512, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_encodings['input_ids'],
                attention_mask=input_encodings['attention_mask'],
                image=processed_image
            )
            
            # Decode generated tokens
            generated_ids = torch.argmax(outputs.logits, dim=-1)
            generated_answer = self.model.tokenizer.decode(
                generated_ids[0], 
                skip_special_tokens=True
            )
        
        return generated_answer
    
    def batch_predict(self, image_paths, questions):
        """
        Batch prediction for multiple images and questions
        
        Args:
            image_paths (List[str]): List of image paths
            questions (List[str]): Corresponding questions
        
        Returns:
            List[str]: Generated answers
        """
        answers = []
        for img_path, question in zip(image_paths, questions):
            answer = self.predict(img_path, question)
            answers.append(answer)
        
        return answers

# Example usage
def run_medical_vqa_inference():
    # Initialize inference utility
    vqa_inference = MedicalVQAInference('medical_vqa_model.pth')
    
    # Single image prediction
    image_path = 'medical_images/xray_001.jpg'
    question = 'What abnormality is present in this chest X-ray?'
    answer = vqa_inference.predict(image_path, question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Batch prediction
    batch_images = [
        'medical_images/xray_001.jpg',
        'medical_images/mri_002.jpg'
    ]
    batch_questions = [
        'What abnormality is present in this chest X-ray?',
        'Describe the brain structure in this MRI scan.'
    ]
    batch_answers = vqa_inference.batch_predict(batch_images, batch_questions)
    
    for img, q, a in zip(batch_images, batch_questions, batch_answers):
        print(f"Image: {img}")
        print(f"Question: {q}")
        print(f"Answer: {a}")
        print("---")

if __name__ == "__main__":
    run_medical_vqa_inference()