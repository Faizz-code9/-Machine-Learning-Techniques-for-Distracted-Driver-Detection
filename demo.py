"""
Live Demonstration Script for Distracted Driver Detection.
Course: UE24CS352A - Machine Learning Mini-Project

Usage:
    python demo.py --image path/to/image.jpg
    python demo.py --data_dir data/train --random
"""

import os
import sys
import glob
import random
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

CLASS_MAPPING = {
    'c0': 'Safe Driving',
    'c1': 'Texting - Right',
    'c2': 'Talking on Phone - Right',
    'c3': 'Texting - Left',
    'c4': 'Talking on Phone - Left',
    'c5': 'Operating Radio',
    'c6': 'Drinking',
    'c7': 'Reaching Behind',
    'c8': 'Hair and Makeup',
    'c9': 'Talking to Passenger'
}

CLASS_NAMES = [CLASS_MAPPING[f'c{i}'] for i in range(10)]

def build_resnet_model(checkpoint_path=None):
    """Loads ResNet-50 model architecture."""
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 10)
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    
    model.eval()
    return model

def predict_single_image(model, img_path):
    """Runs inference on a single image and returns probabilities."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(img_path).convert('RGB')
    tensor_img = transform(raw_img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1).squeeze(0).numpy()
        
    top_class_idx = np.argmax(probs)
    top_class_name = CLASS_NAMES[top_class_idx]
    confidence = probs[top_class_idx] * 100
    
    return raw_img, probs, top_class_idx, top_class_name, confidence

def display_demo_result(raw_img, probs, top_class_idx, top_class_name, confidence):
    """Displays the driver image alongside the class probability distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1.2]})
    
    # 1. Image View with Banner
    ax1.imshow(raw_img)
    ax1.axis('off')
    
    is_safe = (top_class_idx == 0)
    banner_color = '#2E7D32' if is_safe else '#C62828'
    status_text = "🟢 SAFE DRIVING" if is_safe else f"🔴 DISTRACTION: {top_class_name.upper()}"
    
    ax1.set_title(f"{status_text}\nConfidence: {confidence:.1f}%", 
                  fontsize=13, fontweight='bold', color=banner_color, pad=12)
    
    # 2. Probability Bar Chart
    colors_bars = ['#2E7D32' if i == 0 else '#D32F2F' if i == top_class_idx else '#78909C' for i in range(10)]
    bars = ax2.barh(CLASS_NAMES, probs * 100, color=colors_bars, edgecolor='black', linewidth=0.7)
    
    ax2.set_xlim(0, 105)
    ax2.set_xlabel('Probability (%)', fontweight='bold', fontsize=11)
    ax2.set_title('Predicted Probability Distribution Across 10 Classes', fontweight='bold', fontsize=12, pad=12)
    ax2.gca().invert_yaxis()
    
    for bar in bars:
        w = bar.get_width()
        if w > 1.0:
            ax2.text(w + 1.5, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', va='center', fontweight='bold', fontsize=9)
            
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Live Demo for Distracted Driver Detection")
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--data_dir', type=str, default='data/train', help='Path to dataset train directory')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to trained model .pth checkpoint')
    parser.add_argument('--random', action='store_true', help='Pick a random sample from dataset')
    args = parser.parse_args()

    model = build_resnet_model(args.checkpoint)

    target_image = args.image

    if not target_image and args.random:
        all_imgs = glob.glob(os.path.join(args.data_dir, '*', '*.jpg'))
        if all_imgs:
            target_image = random.choice(all_imgs)
            print(f"Randomly selected image: {target_image}")
        else:
            print(f"No images found in {args.data_dir}. Please specify --image <path>.")
            return

    if not target_image:
        # Check for any sample image in data/
        sample_search = glob.glob('data/**/*.jpg', recursive=True)
        if sample_search:
            target_image = sample_search[0]
        else:
            print("Please specify an image path using --image path/to/image.jpg")
            return

    print(f"Running inference on: {target_image}")
    raw_img, probs, top_idx, top_name, conf = predict_single_image(model, target_image)
    
    print("\n" + "="*50)
    print(f"PREDICTION RESULT")
    print("="*50)
    print(f"Top Activity   : {top_name} (c{top_idx})")
    print(f"Confidence     : {conf:.2f}%")
    print("="*50 + "\n")
    
    display_demo_result(raw_img, probs, top_idx, top_name, conf)

if __name__ == '__main__':
    main()
