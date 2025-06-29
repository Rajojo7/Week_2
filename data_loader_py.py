"""
Data loading and preprocessing module for EfficientNet Garbage Classification
Handles dataset loading, analysis, preprocessing, and augmentation
"""

import os
import numpy as np
import pandas as pd
import cv2
import glob
import zipfile
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from google.colab import files, drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

from config import DATASET_CONFIG, AUGMENTATION_CONFIG, PLOT_CONFIG


class GarbageDataLoader:
    """Handles all data loading and preprocessing operations"""
    
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATASET_CONFIG['data_dir']
        self.categories = DATASET_CONFIG['categories']
        self.supported_formats = DATASET_CONFIG['supported_formats']
        self.image_paths = []
        self.labels = []
        
    def upload_and_extract_data(self):
        """Upload and extract dataset (Google Colab specific)"""
        if not COLAB_AVAILABLE:
            print("⚠️ Google Colab not available. Please ensure dataset is in correct directory.")
            return self.data_dir
            
        print("📁 Upload your garbage dataset zip file:")
        uploaded = files.upload()
        
        # Extract the uploaded zip file
        for filename in uploaded.keys():
            if filename.endswith('.zip'):
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall('/content/')
                print(f"✅ Extracted {filename}")
                break
        
        # Find the extracted folder
        possible_paths = [
            '/content/garbage_dataset', 
            '/content/garbage-dataset', 
            '/content/dataset', 
            '/content/data'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.data_dir = path
                print(f"📂 Found dataset at: {self.data_dir}")
                break
        
        return self.data_dir
    
    def mount_drive(self):
        """Mount Google Drive (if using Google Colab)"""
        if COLAB_AVAILABLE:
            try:
                drive.mount('/content/drive')
                print("✅ Google Drive mounted successfully!")
            except Exception as e:
                print(f"⚠️ Could not mount Google Drive: {e}")
        else:
            print("⚠️ Google Colab not available. Skipping drive mount.")
    
    def scan_dataset(self):
        """Scan and catalog all images in the dataset"""
        print("🔍 Scanning dataset...")
        
        self.image_paths = []
        self.labels = []
        category_counts = {}
        total_size = 0
        
        for category in self.categories:
            category_path = os.path.join(self.data_dir, category)
            if os.path.exists(category_path):
                # Find all image files
                image_files = []
                for ext in self.supported_formats:
                    image_files.extend(glob.glob(os.path.join(category_path, ext)))
                    image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
                
                category_counts[category] = len(image_files)
                
                # Store paths and labels
                for img_path in image_files:
                    self.image_paths.append(img_path)
                    self.labels.append(category)
                    
                    # Calculate file size
                    try:
                        total_size += os.path.getsize(img_path)
                    except:
                        pass
            else:
                category_counts[category] = 0
                print(f"⚠️ Warning: {category} folder not found!")
        
        return category_counts, total_size
    
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("🗑️ GARBAGE CLASSIFICATION DATASET ANALYSIS")
        print("=" * 60)
        
        category_counts, total_size = self.scan_dataset()
        total_images = sum(category_counts.values())
        
        if total_images == 0:
            print("❌ No images found! Please check your dataset directory.")
            return category_counts
        
        # Display statistics
        print(f"📊 Dataset Statistics:")
        print(f"   Total Images: {total_images:,}")
        print(f"   Total Categories: {len([c for c in category_counts.values() if c > 0])}")
        print(f"   Dataset Size: {total_size / (1024*1024):.1f} MB")
        print(f"   Average per Category: {total_images/len(self.categories):.0f} images")
        
        print(f"\n🗂️ Images per Category:")
        for category in self.categories:
            count = category_counts[category]
            percentage = (count/total_images*100) if total_images > 0 else 0
            bar = "█" * int(percentage/2) + "░" * (50-int(percentage/2))
            print(f"   {category:>10}: {count:>3} images {bar} {percentage:>5.1f}%")
        
        return category_counts
    
    def visualize_dataset(self, category_counts):
        """Create comprehensive dataset visualizations"""
        plt.style.use(PLOT_CONFIG['style'])
        fig, axes = plt.subplots(2, 2, figsize=PLOT_CONFIG['figsize_large'])
        colors = PLOT_CONFIG['colors']
        
        # 1. Bar chart
        axes[0,0].bar(self.categories, [category_counts[cat] for cat in self.categories], 
                     color=colors, edgecolor='black', linewidth=0.8)
        axes[0,0].set_title('🗑️ Garbage Dataset Distribution', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Waste Categories')
        axes[0,0].set_ylabel('Number of Images')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, (cat, count) in enumerate(zip(self.categories, [category_counts[cat] for cat in self.categories])):
            axes[0,0].text(i, count + 5, str(count), ha='center', fontweight='bold')
        
        # 2. Pie chart
        counts = [category_counts[cat] for cat in self.categories]
        axes[0,1].pie(counts, labels=self.categories, autopct='%1.1f%%', 
                     colors=colors, startangle=90)
        axes[0,1].set_title('📊 Dataset Proportion', fontsize=14, fontweight='bold')
        
        # 3. Data quality assessment
        quality_metrics = ['Images < 100', 'Images 100-300', 'Images 300-500', 'Images > 500']
        quality_counts = [0, 0, 0, 0]
        
        for count in counts:
            if count < 100:
                quality_counts[0] += 1
            elif count < 300:
                quality_counts[1] += 1
            elif count < 500:
                quality_counts[2] += 1
            else:
                quality_counts[3] += 1
        
        axes[1,0].bar(quality_metrics, quality_counts, color='lightcoral', alpha=0.7)
        axes[1,0].set_title('📈 Data Quality Distribution', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('Number of Categories')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Dataset recommendations
        axes[1,1].axis('off')
        recommendations = self._generate_recommendations(counts)
        
        axes[1,1].text(0.1, 0.9, "🎯 Dataset Recommendations:\n\n" + "\n".join(recommendations), 
                      transform=axes[1,1].transAxes, fontsize=10,
                      verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                      facecolor="lightblue", alpha=0.5))
        
        plt.tight_layout()
        plt.show()
    
    def _generate_recommendations(self, counts):
        """Generate dataset-specific recommendations"""
        recommendations = []
        
        min_images = min(counts)
        max_images = max(counts)
        total_images = sum(counts)
        
        if min_images < 200:
            recommendations.append("⚠️ Some categories have < 200 images")
            recommendations.append("   Consider data augmentation")
        
        if max_images - min_images > 200:
            recommendations.append("⚠️ Imbalanced dataset detected")
            recommendations.append("   Consider class balancing")
        
        if total_images > 2000:
            recommendations.append("✅ Good dataset size for deep learning")
        else:
            recommendations.append("⚠️ Small dataset - use transfer learning")
        
        recommendations.append(f"📝 Total training samples: {total_images}")
        recommendations.append(f"📝 Estimated training time: {total_images//80} mins")
        
        return recommendations
    
    def show_sample_images(self, samples_per_category=3):
        """Display sample images from each category"""
        fig, axes = plt.subplots(len(self.categories), samples_per_category, 
                                figsize=(samples_per_category*3, len(self.categories)*2.5))
        
        for i, category in enumerate(self.categories):
            category_path = os.path.join(self.data_dir, category)
            if os.path.exists(category_path):
                # Get image files
                image_files = []
                for ext in self.supported_formats:
                    image_files.extend(glob.glob(os.path.join(category_path, ext)))
                    image_files.extend(glob.glob(os.path.join(category_path, ext.upper())))
                
                for j in range(min(samples_per_category, len(image_files))):
                    try:
                        img = Image.open(image_files[j])
                        if len(axes.shape) > 1:
                            axes[i, j].imshow(img)
                            axes[i, j].set_title(f'{category.upper()} - Sample {j+1}', fontweight='bold')
                            axes[i, j].axis('off')
                        else:
                            axes[i].imshow(img)
                            axes[i].set_title(f'{category.upper()}', fontweight='bold')
                            axes[i].axis('off')
                    except Exception as e:
                        if len(axes.shape) > 1:
                            axes[i, j].text(0.5, 0.5, f'Error loading\n{category}', 
                                           ha='center', va='center', transform=axes[i, j].transAxes)
                            axes[i, j].axis('off')
                        else:
                            axes[i].text(0.5, 0.5, f'Error loading\n{category}', 
                                        ha='center', va='center', transform=axes[i].transAxes)
                            axes[i].axis('off')
            
            # Fill empty slots if multiple samples per category
            if len(axes.shape) > 1:
                for j in range(len(image_files) if len(image_files) < samples_per_category else samples_per_category, samples_per_category):
                    axes[i, j].axis('off')
        
        plt.suptitle('🗑️ Garbage Classification - Sample Images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def preprocess_images(self, img_size=(224, 224)):
        """Preprocess images for EfficientNet training"""
        print(f"🔄 Preprocessing Images to size {img_size}...")
        
        processed_images = []
        processed_labels = []
        corrupted_count = 0
        
        for i, (img_path, label) in enumerate(zip(self.image_paths, self.labels)):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    corrupted_count += 1
                    continue
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize to target size
                img = cv2.resize(img, img_size)
                
                # Normalize pixel values to [0, 1]
                img = img.astype(np.float32) / 255.0
                
                processed_images.append(img)
                processed_labels.append(label)
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(self.image_paths)} images...")
                    
            except Exception as e:
                corrupted_count += 1
                print(f"   Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(processed_images)
        
        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(processed_labels)
        
        print(f"✅ Preprocessing Complete!")
        print(f"   Successfully processed: {len(X)} images")
        print(f"   Corrupted/skipped: {corrupted_count} images")
        print(f"   Final dataset shape: {X.shape}")
        print(f"   Memory usage: {X.nbytes / (1024*1024):.1f} MB")
        
        return X, y, le
    
    def split_dataset(self, X, y):
        """Split dataset into train, validation, and test sets"""
        print("📊 Splitting Dataset...")
        
        # First split: train vs temp (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, 
            test_size=DATASET_CONFIG['test_size'], 
            random_state=DATASET_CONFIG['random_state'], 
            stratify=y
        )
        
        # Second split: val vs test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, 
            test_size=DATASET_CONFIG['val_split'], 
            random_state=DATASET_CONFIG['random_state'], 
            stratify=y_temp
        )
        
        print(f"   Training: {len(X_train)} images ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Validation: {len(X_val)} images ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Testing: {len(X_test)} images ({len(X_test)/len(X)*100:.1f}%)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """Create data generators with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = keras.preprocessing.image.ImageDataGenerator(**AUGMENTATION_CONFIG)
        
        # Validation data generator (no augmentation)
        val_datagen = keras.preprocessing.image.ImageDataGenerator()
        
        # Fit the training generator
        train_datagen.fit(X_train)
        
        # Create generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size, shuffle=True)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size, shuffle=False)
        
        print(f"📦 Data generators created with batch size: {batch_size}")
        
        return train_generator, val_generator


def test_data_loader():
    """Test the data loader functionality"""
    print("🧪 Testing Data Loader...")
    
    # Initialize data loader
    loader = GarbageDataLoader()
    
    # Test dataset analysis (if dataset exists)
    if os.path.exists(loader.data_dir):
        category_counts = loader.analyze_dataset()
        print(f"✅ Found {sum(category_counts.values())} images")
    else:
        print("⚠️ Dataset directory not found. Please upload dataset first.")
    
    print("✅ Data Loader test completed!")


if __name__ == "__main__":
    test_data_loader()