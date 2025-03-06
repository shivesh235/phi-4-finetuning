import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import evaluate
from typing import List, Dict, Any

def calculate_vqa_metrics(predictions: List[str], references: List[str], answer_types: List[str]) -> Dict[str, float]:
    """
    Calculate metrics for VQA task
    
    Args:
        predictions: List of model predictions
        references: List of ground truth answers
        answer_types: List of answer types (CLOSE or OPEN)
        
    Returns:
        Dictionary containing the calculated metrics
    """
    metrics = {}
    
    # Load HuggingFace metrics
    exact_match = evaluate.load("exact_match")
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    
    # Calculate overall metrics
    metrics["exact_match"] = exact_match.compute(predictions=predictions, references=references)["exact_match"]
    metrics["bleu"] = bleu.compute(predictions=predictions, references=[[r] for r in references])["bleu"]
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    metrics["rouge1"] = rouge_scores["rouge1"]
    metrics["rouge2"] = rouge_scores["rouge2"]
    metrics["rougeL"] = rouge_scores["rougeL"]
    
    # Calculate metrics for closed and open-ended questions separately
    close_indices = [i for i, t in enumerate(answer_types) if t == "CLOSE"]
    open_indices = [i for i, t in enumerate(answer_types) if t == "OPEN"]
    
    if close_indices:
        close_preds = [predictions[i] for i in close_indices]
        close_refs = [references[i] for i in close_indices]
        metrics["close_exact_match"] = exact_match.compute(predictions=close_preds, references=close_refs)["exact_match"]
    else:
        metrics["close_exact_match"] = 0.0
    
    if open_indices:
        open_preds = [predictions[i] for i in open_indices]
        open_refs = [references[i] for i in open_indices]
        metrics["open_exact_match"] = exact_match.compute(predictions=open_preds, references=open_refs)["exact_match"]
        metrics["open_bleu"] = bleu.compute(predictions=open_preds, references=[[r] for r in open_refs])["bleu"]
    else:
        metrics["open_exact_match"] = 0.0
        metrics["open_bleu"] = 0.0
    
    return metrics


def calculate_binary_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate metrics for binary (yes/no) questions
    
    Args:
        predictions: List of model predictions (expecting 'yes' or 'no')
        references: List of ground truth answers (expecting 'yes' or 'no')
        
    Returns:
        Dictionary containing the calculated binary classification metrics
    """
    # Normalize predictions and references
    norm_preds = [p.strip().lower() for p in predictions]
    norm_refs = [r.strip().lower() for r in references]
    
    # Convert to binary format
    binary_preds = [1 if p == 'yes' else 0 for p in norm_preds]
    binary_refs = [1 if r == 'yes' else 0 for r in norm_refs]
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(binary_refs, binary_preds),
        "precision": precision_score(binary_refs, binary_preds, zero_division=0),
        "recall": recall_score(binary_refs, binary_preds, zero_division=0),
        "f1": f1_score(binary_refs, binary_preds, zero_division=0)
    }
    
    return metrics