import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models
import timm  # Modern CNN models

# Metrics and utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_fscore_support,
                             roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import label_binarize
from collections import Counter
import time
from tqdm import tqdm
import json

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# =============================================================================
# 1. DATA ANALYSIS AND EXPLORATION
# =============================================================================

class DataAnalyzer:
    """Comprehensive data analysis and visualization"""
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.categories = sorted(os.listdir(data_dir))
        self.categories = [c for c in self.categories if os.path.isdir(os.path.join(data_dir, c))]
        self.analysis_results = {}
        
    def analyze_dataset(self):
        """Perform comprehensive dataset analysis"""
        print("=" * 80)
        print("DATASET ANALYSIS")
        print("=" * 80)
        
        # Count images per category
        category_counts = {}
        all_images = []
        
        for category in self.categories:
            cat_path = os.path.join(self.data_dir, category)
            images = [f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            category_counts[category] = len(images)
            
            for img in images:
                all_images.append({
                    'category': category,
                    'filename': img,
                    'path': os.path.join(cat_path, img)
                })
        
        self.df = pd.DataFrame(all_images)
        self.analysis_results['category_counts'] = category_counts
        
        print(f"\nTotal Images: {len(all_images)}")
        print(f"Number of Categories: {len(self.categories)}")
        print("\nImages per Category:")
        for cat, count in category_counts.items():
            print(f"  {cat:12s}: {count:4d} ({count/len(all_images)*100:5.2f}%)")
        
        # Analyze image properties
        self.analyze_image_properties()
        
        return self.df
    
    def analyze_image_properties(self):
        """Analyze image dimensions, file sizes, and aspect ratios"""
        print("\n" + "=" * 80)
        print("IMAGE PROPERTIES ANALYSIS")
        print("=" * 80)
        
        widths, heights, aspect_ratios, file_sizes = [], [], [], []
        
        # Sample images for analysis (to save time)
        sample_size = min(500, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=42)
        
        for _, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Analyzing images"):
            try:
                img = Image.open(row['path'])
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w/h)
                file_sizes.append(os.path.getsize(row['path']) / 1024)  # KB
            except:
                continue
        
        self.analysis_results['widths'] = widths
        self.analysis_results['heights'] = heights
        self.analysis_results['aspect_ratios'] = aspect_ratios
        self.analysis_results['file_sizes'] = file_sizes
        
        print(f"\nImage Dimensions (from {len(widths)} samples):")
        print(f"  Width  - Mean: {np.mean(widths):.1f}px, Std: {np.std(widths):.1f}px, "
              f"Range: [{np.min(widths)}, {np.max(widths)}]")
        print(f"  Height - Mean: {np.mean(heights):.1f}px, Std: {np.std(heights):.1f}px, "
              f"Range: [{np.min(heights)}, {np.max(heights)}]")
        print(f"  Aspect Ratio - Mean: {np.mean(aspect_ratios):.2f}, Std: {np.std(aspect_ratios):.2f}")
        print(f"  File Size - Mean: {np.mean(file_sizes):.1f} KB, Std: {np.std(file_sizes):.1f} KB")
    
    def create_visualizations(self, save_dir='results/analysis'):
        """Create comprehensive visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # 1. Category Distribution
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        categories = list(self.analysis_results['category_counts'].keys())
        counts = list(self.analysis_results['category_counts'].values())
        colors = sns.color_palette("husl", len(categories))
        
        axes[0].bar(categories, counts, color=colors, edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Category', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribution of Images Across Categories', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (cat, count) in enumerate(zip(categories, counts)):
            axes[0].text(i, count + 10, str(count), ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts, labels=categories, autopct='%1.1f%%', startangle=90, colors=colors,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Percentage Distribution of Categories', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/category_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Image Properties
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Width distribution
        axes[0, 0].hist(self.analysis_results['widths'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(self.analysis_results['widths']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(self.analysis_results["widths"]):.0f}px')
        axes[0, 0].set_xlabel('Width (pixels)', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(self.analysis_results['heights'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(self.analysis_results['heights']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(self.analysis_results["heights"]):.0f}px')
        axes[0, 1].set_xlabel('Height (pixels)', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Aspect ratio distribution
        axes[1, 0].hist(self.analysis_results['aspect_ratios'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(np.mean(self.analysis_results['aspect_ratios']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(self.analysis_results["aspect_ratios"]):.2f}')
        axes[1, 0].set_xlabel('Aspect Ratio (W/H)', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Aspect Ratio Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # File size distribution
        axes[1, 1].hist(self.analysis_results['file_sizes'], bins=50, color='plum', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(self.analysis_results['file_sizes']), color='red', linestyle='--', 
                          linewidth=2, label=f'Mean: {np.mean(self.analysis_results["file_sizes"]):.0f} KB')
        axes[1, 1].set_xlabel('File Size (KB)', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('File Size Distribution', fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/image_properties.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sample images from each category
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, category in enumerate(self.categories):
            cat_df = self.df[self.df['category'] == category]
            sample_img_path = cat_df.sample(1, random_state=42).iloc[0]['path']
            img = Image.open(sample_img_path)
            
            axes[idx].imshow(img)
            axes[idx].set_title(f'{category.upper()}\n({self.analysis_results["category_counts"][category]} images)', 
                              fontsize=12, fontweight='bold')
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualizations saved to: {save_dir}/")
        
    def save_statistics(self, save_dir='results/analysis'):
        """Save detailed statistics to JSON"""
        os.makedirs(save_dir, exist_ok=True)
        
        stats = {
            'total_images': len(self.df),
            'num_categories': len(self.categories),
            'categories': self.categories,
            'category_distribution': self.analysis_results['category_counts'],
            'image_properties': {
                'width': {
                    'mean': float(np.mean(self.analysis_results['widths'])),
                    'std': float(np.std(self.analysis_results['widths'])),
                    'min': int(np.min(self.analysis_results['widths'])),
                    'max': int(np.max(self.analysis_results['widths']))
                },
                'height': {
                    'mean': float(np.mean(self.analysis_results['heights'])),
                    'std': float(np.std(self.analysis_results['heights'])),
                    'min': int(np.min(self.analysis_results['heights'])),
                    'max': int(np.max(self.analysis_results['heights']))
                },
                'aspect_ratio': {
                    'mean': float(np.mean(self.analysis_results['aspect_ratios'])),
                    'std': float(np.std(self.analysis_results['aspect_ratios']))
                },
                'file_size_kb': {
                    'mean': float(np.mean(self.analysis_results['file_sizes'])),
                    'std': float(np.std(self.analysis_results['file_sizes']))
                }
            }
        }
        
        with open(f'{save_dir}/dataset_statistics.json', 'w') as f:
            json.dump(stats, f, indent=4)
        
        print(f"Statistics saved to: {save_dir}/dataset_statistics.json")

# =============================================================================
# 2. DATA PREPROCESSING AND AUGMENTATION
# =============================================================================

class GarbageDataset(Dataset):
    """Custom Dataset for Garbage Classification"""
    
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_map = {cat: idx for idx, cat in enumerate(sorted(dataframe['category'].unique()))}
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[self.df.iloc[idx]['category']]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms(image_size=224, augment=True):
    """Define data transforms for preprocessing and augmentation"""
    
    if augment:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation/Test transforms without augmentation
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """Split data into train, validation, and test sets"""
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['category']
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state, stratify=train_val_df['category']
    )
    
    print("\n" + "=" * 80)
    print("DATA SPLIT")
    print("=" * 80)
    print(f"Train set: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Validation set: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"Test set: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df

# =============================================================================
# 3. MODEL ARCHITECTURES
# =============================================================================

class CustomCNN(nn.Module):
    """Custom CNN architecture from scratch"""
    
    def __init__(self, num_classes=6):
        super(CustomCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def get_model(model_name, num_classes=6, pretrained=True):
    """Get model by name"""
    
    if model_name == 'custom_cnn':
        model = CustomCNN(num_classes=num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
        
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model

# =============================================================================
# 4. TRAINING AND EVALUATION
# =============================================================================

class ModelTrainer:
    """Model training and evaluation class"""
    
    def __init__(self, model, model_name, device, save_dir='results/models'):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/len(pbar), 'acc': 100.*correct/total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=30, lr=0.001, patience=7):
        """Full training loop with early stopping"""
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                         patience=3)
        
        best_val_acc = 0.0
        epochs_no_improve = 0
        best_model_path = f'{self.save_dir}/{self.model_name}_best.pth'
        
        print(f"\n{'='*80}")
        print(f"Training {self.model_name.upper()}")
        print(f"{'='*80}")
        print(f"Epochs: {epochs} | Learning Rate: {lr} | Device: {self.device}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"✓ Best model saved with validation accuracy: {best_val_acc:.2f}%")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
            
            print("-" * 80)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Load best model
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.history, training_time
    
    def evaluate(self, test_loader, class_names):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs,
            'class_names': class_names
        }
        
        print(f"\n{'='*80}")
        print(f"TEST SET EVALUATION - {self.model_name.upper()}")
        print(f"{'='*80}")
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"{'='*80}\n")
        
        # Classification report
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        
        return results

# =============================================================================
# 5. VISUALIZATION AND COMPARISON
# =============================================================================

class ResultVisualizer:
    """Visualize and compare model results"""
    
    def __init__(self, save_dir='results/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def plot_training_history(self, histories, model_names):
        """Plot training curves for all models"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Loss curves
        for history, name in zip(histories, model_names):
            axes[0].plot(history['train_loss'], label=f'{name} (Train)', linewidth=2)
            axes[0].plot(history['val_loss'], label=f'{name} (Val)', linestyle='--', linewidth=2)
        
        axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Accuracy curves
        for history, name in zip(histories, model_names):
            axes[1].plot(history['train_acc'], label=f'{name} (Train)', linewidth=2)
            axes[1].plot(history['val_acc'], label=f'{name} (Val)', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to: {self.save_dir}/training_curves.png")
    
    def plot_confusion_matrix(self, results, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(results['labels'], results['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=results['class_names'],
                   yticklabels=results['class_names'],
                   cbar_kws={'label': 'Count'})
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.title(f'Confusion Matrix - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, results, model_name):
        """Plot ROC curves for each class"""
        n_classes = len(results['class_names'])
        
        # Binarize labels
        y_test_bin = label_binarize(results['labels'], classes=range(n_classes))
        y_score = np.array(results['probabilities'])
        
        # Compute ROC curve and AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot
        plt.figure(figsize=(12, 10))
        colors = sns.color_palette("husl", n_classes)
        
        for i, color in enumerate(colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{results["class_names"][i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curves - {model_name.upper()}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/roc_curves_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, all_results, model_names, training_times):
        """Create comprehensive model comparison"""
        
        # Prepare data
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy (%)': [r['accuracy']*100 for r in all_results],
            'Precision': [r['precision'] for r in all_results],
            'Recall': [r['recall'] for r in all_results],
            'F1-Score': [r['f1_score'] for r in all_results],
            'Training Time (min)': [t/60 for t in training_times]
        })
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Accuracy comparison
        colors = sns.color_palette("husl", len(model_names))
        axes[0, 0].bar(model_names, metrics_df['Accuracy (%)'], color=colors, edgecolor='black', linewidth=1.2)
        axes[0, 0].set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['Accuracy (%)']):
            axes[0, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. Precision, Recall, F1 comparison
        x = np.arange(len(model_names))
        width = 0.25
        axes[0, 1].bar(x - width, metrics_df['Precision'], width, label='Precision', edgecolor='black')
        axes[0, 1].bar(x, metrics_df['Recall'], width, label='Recall', edgecolor='black')
        axes[0, 1].bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', edgecolor='black')
        axes[0, 1].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Precision, Recall, F1-Score Comparison', fontsize=12, fontweight='bold')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(model_names, rotation=45)
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Training time comparison
        axes[1, 0].bar(model_names, metrics_df['Training Time (min)'], color=colors, edgecolor='black', linewidth=1.2)
        axes[1, 0].set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Training Time Comparison', fontsize=12, fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(metrics_df['Training Time (min)']):
            axes[1, 0].text(i, v + 0.1, f'{v:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency score (accuracy / time)
        efficiency = metrics_df['Accuracy (%)'] / metrics_df['Training Time (min)']
        axes[1, 1].bar(model_names, efficiency, color=colors, edgecolor='black', linewidth=1.2)
        axes[1, 1].set_ylabel('Efficiency (Acc% per minute)', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Model Efficiency (Accuracy / Training Time)', fontsize=12, fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison table
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(metrics_df.to_string(index=False))
        print("=" * 80 + "\n")
        
        metrics_df.to_csv(f'{self.save_dir}/model_comparison.csv', index=False)
        print(f"Comparison table saved to: {self.save_dir}/model_comparison.csv")

# =============================================================================
# 6. MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    # Configuration
    DATA_DIR = r'C:\Users\alper\PROJELER\Capstone2\Garbage classification'
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    # Models to train
    MODELS = ['custom_cnn', 'resnet50', 'efficientnet_b0', 'mobilenet_v3']
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE GARBAGE CLASSIFICATION PROJECT")
    print("=" * 80)
    print(f"Dataset Directory: {DATA_DIR}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Models: {', '.join([m.upper() for m in MODELS])}")
    print("=" * 80 + "\n")
    
    # Step 1: Data Analysis
    print("STEP 1: DATA ANALYSIS")
    analyzer = DataAnalyzer(DATA_DIR)
    df = analyzer.analyze_dataset()
    analyzer.create_visualizations()
    analyzer.save_statistics()
    
    # Step 2: Data Preparation
    print("\nSTEP 2: DATA PREPARATION")
    train_df, val_df, test_df = prepare_data(df)
    
    # Get transforms
    train_transform, val_transform = get_transforms(IMAGE_SIZE, augment=True)
    
    # Create datasets
    train_dataset = GarbageDataset(train_df, transform=train_transform)
    val_dataset = GarbageDataset(val_df, transform=val_transform)
    test_dataset = GarbageDataset(test_df, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # Get class names
    class_names = sorted(df['category'].unique())
    
    print(f"\nData loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Step 3: Train Models
    print("\n" + "=" * 80)
    print("STEP 3: MODEL TRAINING")
    print("=" * 80 + "\n")
    
    all_histories = []
    all_results = []
    training_times = []
    
    for model_name in MODELS:
        print(f"\n{'#'*80}")
        print(f"# TRAINING MODEL: {model_name.upper()}")
        print(f"{'#'*80}\n")
        
        # Get model
        model = get_model(model_name, num_classes=len(class_names))
        
        # Create trainer
        trainer = ModelTrainer(model, model_name, device)
        
        # Train
        history, train_time = trainer.train(train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE)
        all_histories.append(history)
        training_times.append(train_time)
        
        # Evaluate
        results = trainer.evaluate(test_loader, class_names)
        all_results.append(results)
        
        print("\n" + "=" * 80 + "\n")
    
    # Step 4: Visualization and Comparison
    print("STEP 4: RESULTS VISUALIZATION AND COMPARISON")
    visualizer = ResultVisualizer()
    
    # Plot training curves
    visualizer.plot_training_history(all_histories, MODELS)
    
    # Plot confusion matrices and ROC curves for each model
    for results, model_name in zip(all_results, MODELS):
        visualizer.plot_confusion_matrix(results, model_name)
        visualizer.plot_roc_curves(results, model_name)
    
    # Compare all models
    visualizer.compare_models(all_results, MODELS, training_times)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nAll results have been saved to the 'results/' directory:")
    print("  - results/analysis/        : Dataset analysis and statistics")
    print("  - results/models/          : Trained model checkpoints")
    print("  - results/visualizations/  : All plots and comparisons")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
