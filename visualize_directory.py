import matplotlib.pyplot as plt
import os
import random
from PIL import Image


def plot_emotion_distribution(base_path):
    """
    Counts the number of images in each emotion subfolder within the 'train' directory 
    and displays a bar chart.
    """
    train_dir = os.path.join(base_path, 'train')
    if not os.path.exists(train_dir):
        print(f"Error: Train directory not found at {train_dir}")
        return

    # Get list of emotion classes (subdirectories)
    emotions = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    emotions.sort()
    
    print(f"Classes found: {emotions}")
    
    counts = []
    for emotion in emotions:
        emotion_path = os.path.join(train_dir, emotion)
        # Count files (assuming images)
        count = len([f for f in os.listdir(emotion_path) if os.path.isfile(os.path.join(emotion_path, f))])
        counts.append(count)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    bars = plt.bar(emotions, counts, color='skyblue')
    plt.title('Distribution of Emotions in Training Set')
    plt.xlabel('Emotion')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Add count labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def show_sample_images(base_path, n=5):
    """
    Displays a grid of n random images for each emotion class from the training set.
    """
    train_dir = os.path.join(base_path, 'train')
    if not os.path.exists(train_dir):
        return

    emotions = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    emotions.sort()
    
    fig, axes = plt.subplots(len(emotions), n, figsize=(2*n, 2*len(emotions)))
    fig.suptitle('Sample Images from Training Set', fontsize=16)
    
    for i, emotion in enumerate(emotions):
        emotion_path = os.path.join(train_dir, emotion)
        all_images = [f for f in os.listdir(emotion_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Select random samples
        if len(all_images) >= n:
            samples = random.sample(all_images, n)
        else:
            samples = all_images
            
        for j in range(n):
            ax = axes[i, j]
            
            if j < len(samples):
                image_name = samples[j]
                img_path = os.path.join(emotion_path, image_name)
                try:
                    img = Image.open(img_path)
                    ax.imshow(img, cmap='gray')
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
            
            ax.axis('off')
            # Add row label
            if j == 0:
                ax.set_title(emotion, loc='left', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


if __name__ == "__main__":
    # Base path to the dataset
    base_path = "data/raw"
    
    if os.path.exists(base_path):
        print("Visualizing dataset structure...")
        plot_emotion_distribution(base_path)
        show_sample_images(base_path)
    else:
        print(f"Directory {base_path} does not exist. Please run the download script first.")
