import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import itertools
from torch.optim import Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import LinearLR

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

# ==================== MODEL INITIALIZATION ====================
# Initialize tokenizer and model
model_name = "FacebookAI/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16)

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

# Create datasets and dataloaders
train_dataset = MBTIDataset(train_texts, train_labels, tokenizer)
val_dataset = MBTIDataset(val_texts, val_labels, tokenizer)

# ==================== HELPER FUNCTIONS ====================
def get_optimizer(optimizer_name, model_params, lr):
    """Initialize optimizer based on name"""
    if optimizer_name == 'adam':
        return Adam(model_params, lr=lr)
    elif optimizer_name == 'adamw':
        return AdamW(model_params, lr=lr)
    elif optimizer_name == 'rmsprop':
        return RMSprop(model_params, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def mbti_to_string(label):
    """Convert numeric label back to MBTI string"""
    return le.inverse_transform([label.item()])[0]

# ==================== GRID SEARCH CONFIGURATION ====================
# Define hyperparameter grid
optimizers = ['adam', 'adamw', 'rmsprop']
learning_rates = [1e-5, 3e-5, 5e-5]
batch_sizes = [16, 32, 64]
warmup_steps = [0, 100, 500]

# Store results
results = []

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== GRID SEARCH IMPLEMENTATION ====================
total_combinations = len(optimizers) * len(learning_rates) * len(batch_sizes) * len(warmup_steps)
print(f"Starting grid search with {total_combinations} combinations...")

# Generate all combinations of hyperparameters
param_combinations = list(itertools.product(optimizers, learning_rates, batch_sizes, warmup_steps))

for i, (opt_name, lr, batch_size, warmup) in enumerate(param_combinations):
    print(f"\n[{i+1}/{total_combinations}] Testing combination: "
          f"Optimizer={opt_name}, LR={lr}, Batch Size={batch_size}, Warmup={warmup}")
    
    # Create new model instance for each combination
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=16)
    model.to(device)
    
    # Create data loaders with current batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize optimizer and scheduler
    optimizer = get_optimizer(opt_name, model.parameters(), lr)
    
    # Initialize learning rate scheduler for warmup
    if warmup > 0:
        scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
    else:
        scheduler = None
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop (3 epochs only)
    best_val_accuracy = 0.0
    
    for epoch in range(3):  # Fixed to 3 epochs as requested
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        train_progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}", leave=False)
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
            if scheduler and batch_idx < warmup:
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
            
            # Update progress bar
            train_progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)
            for inputs, labels in val_progress_bar:
                # Move data to device
                inputs = {key: val.to(device) for key, val in inputs.items()}
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(**inputs)
                logits = outputs.logits
                _, preds = torch.max(logits, dim=1)
                
                # Update metrics
                val_correct += torch.sum(preds == labels).item()
                val_total += labels.size(0)
                
                val_progress_bar.set_postfix({"Val Acc": f"{val_correct/val_total:.4f}"})
        
        # Calculate epoch metrics
        train_accuracy = correct / total
        val_accuracy = val_correct / val_total
        
        print(f"Epoch {epoch+1} - Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Track best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
    
    # Store results for this combination
    result = {
        'optimizer': opt_name,
        'learning_rate': lr,
        'batch_size': batch_size,
        'warmup_steps': warmup,
        'best_val_accuracy': best_val_accuracy
    }
    results.append(result)
    
    print(f"Best validation accuracy for this combination: {best_val_accuracy:.4f}")

# ==================== RESULTS ANALYSIS ====================
# Sort results by validation accuracy
results_sorted = sorted(results, key=lambda x: x['best_val_accuracy'], reverse=True)

print("\n" + "="*80)
print("GRID SEARCH RESULTS (ranked by validation accuracy)")
print("="*80)

for i, result in enumerate(results_sorted):
    print(f"{i+1:2d}. Val Acc: {result['best_val_accuracy']:.4f} | "
          f"Opt: {result['optimizer']:>7} | "
          f"LR: {result['learning_rate']:>7} | "
          f"Batch: {result['batch_size']:>2} | "
          f"Warmup: {result['warmup_steps']:>3}")

# Display top 5 configurations
print("\n" + "="*50)
print("TOP 5 CONFIGURATIONS")
print("="*50)
for i in range(min(5, len(results_sorted))):
    result = results_sorted[i]
    print(f"#{i+1}: {result['best_val_accuracy']:.4f} - "
          f"{result['optimizer']} LR={result['learning_rate']} "
          f"BS={result['batch_size']} Warmup={result['warmup_steps']}")

# Save results to CSV
results_df = pd.DataFrame(results_sorted)
print(results_df)