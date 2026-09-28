"""
Configuration constants for the Distracted Driver Detection project.
"""

import torch

IMG_SIZE_FLAT = 64
IMG_SIZE_CNN = 224
NUM_CLASSES = 10
BATCH_SIZE = 32
NUM_EPOCHS_TRADITIONAL = 50
NUM_EPOCHS_NN = 30
NUM_EPOCHS_CNN = 15
LEARNING_RATE_NN = 1e-3
LEARNING_RATE_CNN = 1e-4
VAL_SPLIT = 0.2
RANDOM_SEED = 42

CLASS_NAMES = [
    'Safe Driving',
    'Texting - Right',
    'Talking on Phone - Right',
    'Texting - Left',
    'Talking on Phone - Left',
    'Operating Radio',
    'Drinking',
    'Reaching Behind',
    'Hair and Makeup',
    'Talking to Passenger'
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
