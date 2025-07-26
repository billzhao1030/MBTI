import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
from sklearn.metrics import classification_report

# ==================== DATA LOADING AND PREPROCESSING ====================
# Load the cleaned MBTI dataset
data = pd.read_csv('./mbti_cleaned.csv')

# Encode MBTI personality types into numeric labels (0-15 for 16 types)
le = LabelEncoder()
data['label'] = le.fit_transform(data['type'])  # Convert MBTI types to integer labels

# Split data into train (60%), validation (20%), and test (20%) sets
# First split: 80% train+val, 20% test
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    data['cleaned_posts'], data['label'], test_size=0.2, stratify=data['label'], random_state=42
)

# Second split: Split train+val into 60% train and 20% validation (of total)
# 0.25 of 0.8 = 0.2 of total data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.25, stratify=train_val_labels, random_state=42
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")
print(f"Test samples: {len(test_texts)}")

# ==================== DATASET CLASS ====================
class MBTIDataset(Dataset):
    """
    Custom Dataset class for MBTI text classification
    Handles tokenization and data preparation for PyTorch DataLoader
    """
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts.reset_index(drop=True)  # Reset index to avoid indexing issues
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Get text and label for the given index
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_len
        )
        
        # Remove batch dimension from tensors
        inputs = {key: val.squeeze(0) for key, val in encoding.items()}
        return inputs, torch.tensor(label, dtype=torch.long)

# ==================== HELPER FUNCTIONS ====================
def mbti_to_string(label):
    """Convert numeric label back to MBTI string"""
    return le.inverse_transform([label.item()])[0]

def validate_model(model, val_loader, criterion, device, le):
    """Validate the model and return comprehensive metrics"""
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    val_partial_accuracy = 0
    val_correct_ie = 0
    val_correct_sn = 0
    val_correct_tf = 0
    val_correct_pj = 0

    all_true_labels = []
    all_pred_labels = []

    val_progress_bar = tqdm(val_loader, desc="Validation")
    with torch.no_grad():
        for batch in val_progress_bar:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            val_loss += loss.item()

            # Accuracy calculations
            _, preds = torch.max(logits, dim=1)
            val_correct += torch.sum(preds == labels).item()
            val_total += labels.size(0)

            # Keep track of true and predicted labels for classification report
            all_true_labels.extend([mbti_to_string(label) for label in labels])
            all_pred_labels.extend([mbti_to_string(pred) for pred in preds])

            # Calculate dimension-wise accuracy and partial credit
            for i in range(len(labels)):
                true_label = mbti_to_string(labels[i])
                pred_label = mbti_to_string(preds[i])

                # Check individual dimensions
                val_correct_ie += int(true_label[0] == pred_label[0])
                val_correct_sn += int(true_label[1] == pred_label[1])
                val_correct_tf += int(true_label[2] == pred_label[2])
                val_correct_pj += int(true_label[3] == pred_label[3])

                # Partial accuracy (0.25 per correct dimension)
                partial_accuracy = (
                    (true_label[0] == pred_label[0]) +
                    (true_label[1] == pred_label[1]) +
                    (true_label[2] == pred_label[2]) +
                    (true_label[3] == pred_label[3])
                ) / 4.0
                val_partial_accuracy += partial_accuracy

            # Update validation progress bar
            val_progress_bar.set_postfix({
                "Val Loss": f"{loss.item():.4f}",
                "Val Accuracy": f"{(val_correct / val_total):.4f}",
                "Val IE Acc": f"{(val_correct_ie / val_total):.4f}",
                "Val SN Acc": f"{(val_correct_sn / val_total):.4f}",
                "Val TF Acc": f"{(val_correct_tf / val_total):.4f}",
                "Val PJ Acc": f"{(val_correct_pj / val_total):.4f}",
                "Partial Acc": f"{(val_partial_accuracy / val_total):.4f}"
            })

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_total
    avg_val_partial_accuracy = val_partial_accuracy / val_total

    # Print validation metrics
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, Partial Accuracy: {avg_val_partial_accuracy:.4f}")
    print(f"Val IE Acc: {val_correct_ie / val_total:.4f}, Val SN Acc: {val_correct_sn / val_total:.4f}, Val TF Acc: {val_correct_tf / val_total:.4f}, Val PJ Acc: {val_correct_pj / val_total:.4f}")

    # Generate classification report with MBTI labels
    print("\nClassification Report:")
    print(classification_report(all_true_labels, all_pred_labels))
    
    return val_accuracy, avg_val_loss

def test_model(model, test_loader, criterion, device, le):
    """Test the model on test set"""
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0
    test_partial_accuracy = 0
    test_correct_ie = 0
    test_correct_sn = 0
    test_correct_tf = 0
    test_correct_pj = 0

    all_true_labels = []
    all_pred_labels = []

    test_progress_bar = tqdm(test_loader, desc="Testing")
    with torch.no_grad():
        for batch in test_progress_bar:
            inputs, labels = batch
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            test_loss += loss.item()

            # Accuracy calculations
            _, preds = torch.max(logits, dim=1)
            test_correct += torch.sum(preds == labels).item()
            test_total += labels.size(0)

            # Keep track of true and predicted labels for classification report
            all_true_labels.extend([mbti_to_string(label) for label in labels])
            all_pred_labels.extend([mbti_to_string(pred) for pred in preds])

            # Calculate dimension-wise accuracy and partial credit
            for i in range(len(labels)):
                true_label = mbti_to_string(labels[i])
                pred_label = mbti_to_string(preds[i])

                # Check individual dimensions
                test_correct_ie += int(true_label[0] == pred_label[0])
                test_correct_sn += int(true_label[1] == pred_label[1])
                test_correct_tf += int(true_label[2] == pred_label[2])
                test_correct_pj += int(true_label[3] == pred_label[3])

                # Partial accuracy (0.25 per correct dimension)
                partial_accuracy = (
                    (true_label[0] == pred_label[0]) +
                    (true_label[1] == pred_label[1]) +
                    (true_label[2] == pred_label[2]) +
                    (true_label[3] == pred_label[3])
                ) / 4.0
                test_partial_accuracy += partial_accuracy

            # Update test progress bar
            test_progress_bar.set_postfix({
                "Test Loss": f"{loss.item():.4f}",
                "Test Accuracy": f"{(test_correct / test_total):.4f}",
                "Test IE Acc": f"{(test_correct_ie / test_total):.4f}",
                "Test SN Acc": f"{(test_correct_sn / test_total):.4f}",
                "Test TF Acc": f"{(test_correct_tf / test_total):.4f}",
                "Test PJ Acc": f"{(test_correct_pj / test_total):.4f}",
                "Partial Acc": f"{(test_partial_accuracy / test_total):.4f}"
            })

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = test_correct / test_total
    avg_test_partial_accuracy = test_partial_accuracy / test_total

    # Print test metrics
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Partial Accuracy: {avg_test_partial_accuracy:.4f}")
    print(f"Test IE Acc: {test_correct_ie / test_total:.4f}, Test SN Acc: {test_correct_sn / test_total:.4f}, Test TF Acc: {test_correct_tf / test_total:.4f}, Test PJ Acc: {test_correct_pj / test_total:.4f}")

    # Generate classification report with MBTI labels
    print("\nTest Classification Report:")
    print(classification_report(all_true_labels, all_pred_labels))
    
    return test_accuracy, avg_test_loss

# ==================== TRAINING FUNCTION ====================
def train_model(model_name, model_output_dir):
    """Train a single model with best hyperparameters"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16)
    
    # Create datasets and dataloaders with best hyperparameters
    train_dataset = MBTIDataset(train_texts, train_labels, tokenizer)
    val_dataset = MBTIDataset(val_texts, val_labels, tokenizer)
    test_dataset = MBTIDataset(test_texts, test_labels, tokenizer)
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)
    
    # Initialize optimizer and loss function with best hyperparameters
    learning_rate = 5e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Learning rate scheduler for warmup
    warmup_steps = 100
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    
    # Training parameters
    num_epochs = 10
    best_val_accuracy = 0.0
    best_checkpoint_path = os.path.join(model_output_dir, f"best_{model_name.replace('/', '_')}_checkpoint.pth")
    
    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch_idx, (inputs, labels) in enumerate(train_progress_bar):
            # Move data to device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply warmup scheduler
            if batch_idx < warmup_steps:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            # Update progress bar
            train_progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy": f"{(correct / total):.4f}"
            })
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        print(f"Training Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
        
        # Validation phase
        val_accuracy, val_loss = validate_model(model, val_loader, criterion, device, le)
        
        # Save best checkpoint
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            # Save model checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
            }, best_checkpoint_path)
            print(f"New best model saved with validation accuracy: {val_accuracy:.4f}")
    
    print(f"\nBest validation accuracy for {model_name}: {best_val_accuracy:.4f}")
    
    # Load best checkpoint for final testing
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test the model
    print(f"\n{'='*60}")
    print(f"Testing {model_name} with best checkpoint")
    print(f"{'='*60}")
    test_accuracy, test_loss = test_model(model, test_loader, criterion, device, le)
    
    return model, best_val_accuracy, test_accuracy, best_checkpoint_path

# ==================== MAIN TRAINING LOOP ====================
if __name__ == "__main__":
    # Train both models
    models_to_train = [
        ("distilroberta-base", "./distilroberta_results"),
        ("FacebookAI/roberta-base", "./roberta_results")
    ]
    
    results = {}
    
    for model_name, output_dir in models_to_train:
        try:
            model, best_val_acc, test_acc, checkpoint_path = train_model(model_name, output_dir)
            results[model_name] = {
                'best_val_accuracy': best_val_acc,
                'test_accuracy': test_acc,
                'checkpoint_path': checkpoint_path
            }
            print(f"\n{model_name} Training Complete!")
            print(f"Best Validation Accuracy: {best_val_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Checkpoint saved at: {checkpoint_path}")
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            continue
    
    # Final comparison
    print(f"\n{'='*80}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*80}")
    
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Best Validation Accuracy: {metrics['best_val_accuracy']:.4f}")
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Checkpoint: {metrics['checkpoint_path']}")
        print()