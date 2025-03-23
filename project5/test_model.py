import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from neural_net import NeuralNet  # Import our model architecture

def main():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the test dataset
    test_dataset = torchvision.datasets.MNIST('./data', train=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    
    # Create data loader for first 10 examples (no shuffling to keep order)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)
    
    # Load the trained model
    model = NeuralNet().to(device)
    model.load_state_dict(torch.load('mnist_model.pth'))
    
    # Set model to evaluation mode
    model.eval()
    
    # Get first batch (10 examples)
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.exp(outputs)  # Convert from log_softmax to probabilities
        predictions = outputs.max(1)[1]
    
    # Print results for all 10 examples
    print("\nPredictions for first 10 test examples:")
    print("----------------------------------------")
    for i in range(10):
        probs = probabilities[i].cpu().numpy()
        pred = predictions[i].item()
        true_label = labels[i].item()
        
        print(f"\nExample {i+1}:")
        print(f"Output values: {', '.join([f'{p:.2f}' for p in probs])}")
        print(f"Predicted digit: {pred}")
        print(f"Correct label: {true_label}")
        print(f"Correct: {'✓' if pred == true_label else '✗'}")
    
    # Plot first 9 examples in a 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    
    for i, ax in enumerate(axes.flat[:9]):
        # Get the image and remove normalization for display
        img = images[i].cpu().squeeze()
        
        # Plot the image
        ax.imshow(img, cmap='gray')
        
        # Add prediction as title
        pred = predictions[i].item()
        true_label = labels[i].item()
        title = f'Pred: {pred}'
        if pred != true_label:
            title += f' (True: {true_label})'
        ax.set_title(title)
        
        # Remove axes
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions.png')
    plt.close()

if __name__ == "__main__":
    main() 