"""
Configuration file for EfficientNet Garbage Classification Project
Contains all hyperparameters, paths, and model configurations
"""

import os

# Dataset Configuration
DATASET_CONFIG = {
    'categories': ['glass', 'cardboard', 'plastic', 'metal', 'paper', 'trash'],
    'data_dir': '/content/garbage_dataset',
    'supported_formats': ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff'],
    'test_size': 0.3,
    'val_split': 0.5,
    'random_state': 42
}

# EfficientNet Model Configurations
EFFICIENTNET_CONFIGS = {
    'B0': {
        'input_size': (224, 224),
        'batch_size': 32,
        'initial_lr': 0.001,
        'fine_tune_lr': 0.0001,
        'fine_tune_at': 200,
        'parameters': 5.3,  # Million parameters
        'memory_usage': 'Low'
    },
    'B1': {
        'input_size': (240, 240),
        'batch_size': 24,
        'initial_lr': 0.0009,
        'fine_tune_lr': 0.00009,
        'fine_tune_at': 250,
        'parameters': 7.8,
        'memory_usage': 'Medium'
    },
    'B2': {
        'input_size': (260, 260),
        'batch_size': 20,
        'initial_lr': 0.0008,
        'fine_tune_lr': 0.00008,
        'fine_tune_at': 280,
        'parameters': 9.2,
        'memory_usage': 'Medium'
    },
    'B3': {
        'input_size': (300, 300),
        'batch_size': 16,
        'initial_lr': 0.0008,
        'fine_tune_lr': 0.00008,
        'fine_tune_at': 300,
        'parameters': 12.0,
        'memory_usage': 'High'
    },
    'B4': {
        'input_size': (380, 380),
        'batch_size': 12,
        'initial_lr': 0.0007,
        'fine_tune_lr': 0.00007,
        'fine_tune_at': 350,
        'parameters': 19.0,
        'memory_usage': 'Very High'
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'initial_epochs': 35,
    'fine_tune_epochs': 15,
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    'reduce_lr_factor': 0.2,
    'min_lr': 1e-8,
    'lr_decay_rate': 0.95
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 25,
    'width_shift_range': 0.15,
    'height_shift_range': 0.15,
    'shear_range': 0.1,
    'zoom_range': 0.15,
    'horizontal_flip': True,
    'vertical_flip': False,
    'brightness_range': [0.85, 1.15],
    'channel_shift_range': 0.1,
    'fill_mode': 'nearest'
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'dropout_rates': [0.3, 0.5, 0.3],
    'hidden_units': [512, 256],
    'activation': 'relu',
    'final_activation': 'softmax',
    'use_batch_norm': True,
    'optimizer': 'adam',
    'loss': 'sparse_categorical_crossentropy',
    'metrics': ['accuracy', 'top_2_accuracy']
}

# Visualization Configuration
PLOT_CONFIG = {
    'figsize_large': (16, 12),
    'figsize_medium': (12, 8),
    'figsize_small': (8, 6),
    'colors': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'],
    'style': 'default',
    'dpi': 100,
    'save_format': 'png'
}

# File Paths Configuration
PATHS_CONFIG = {
    'models_dir': '/content/models',
    'results_dir': '/content/results',
    'logs_dir': '/content/logs',
    'temp_dir': '/content/temp',
    'model_checkpoint_template': '/content/best_efficientnet_{variant}_garbage_model.h5',
    'final_model_template': '/content/final_efficientnet_{variant}_garbage_classifier.h5'
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'excellent': 0.90,
    'very_good': 0.85,
    'good': 0.80,
    'needs_improvement': 0.80
}

# Recycling Tips
RECYCLING_TIPS = {
    'paper': "♻️ Recyclable! Remove any plastic coating first.",
    'cardboard': "♻️ Recyclable! Flatten boxes to save space.",
    'plastic': "♻️ Check recycling number. Clean containers first.",
    'glass': "♻️ Recyclable! Remove caps and clean thoroughly.",
    'metal': "♻️ Recyclable! Clean cans and remove labels.",
    'trash': "🗑️ General waste. Consider if any parts can be recycled."
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_handler': True,
    'console_handler': True
}

def get_model_config(variant='B0'):
    """Get configuration for specific EfficientNet variant"""
    return EFFICIENTNET_CONFIGS.get(variant, EFFICIENTNET_CONFIGS['B0'])

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        PATHS_CONFIG['models_dir'],
        PATHS_CONFIG['results_dir'],
        PATHS_CONFIG['logs_dir'],
        PATHS_CONFIG['temp_dir']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("📁 Project directories created successfully!")

def validate_config():
    """Validate configuration settings"""
    # Check if all required categories are defined
    assert len(DATASET_CONFIG['categories']) > 0, "No categories defined!"
    
    # Check if EfficientNet configs are valid
    for variant, config in EFFICIENTNET_CONFIGS.items():
        assert 'input_size' in config, f"Missing input_size for {variant}"
        assert 'batch_size' in config, f"Missing batch_size for {variant}"
        assert 'initial_lr' in config, f"Missing initial_lr for {variant}"
    
    print("✅ Configuration validation passed!")

if __name__ == "__main__":
    validate_config()
    create_directories()
    print("🎯 Configuration loaded successfully!")