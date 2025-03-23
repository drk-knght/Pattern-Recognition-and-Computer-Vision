import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from neural_net import NeuralNet
import cv2
import torchvision

class NetworkAnalyzer:
    def __init__(self, model_path: str):
        """
        Initialize the NetworkAnalyzer with a path to the trained model.
        
        Args:
            model_path (str): Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.first_layer_weights = None
    
    def load_model(self) -> None:
        """Load the trained model from the specified path."""
        try:
            # Create a new model instance
            self.model = NeuralNet()
            # Load the state dictionary
            self.model.load_state_dict(torch.load(self.model_path))
            # Set to evaluation mode
            self.model.eval()
            print("Model structure:")
            print(self.model)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def extract_first_layer_weights(self) -> None:
        """Extract weights from the first convolutional layer."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        try:
            self.first_layer_weights = self.model.conv1.weight
            print("\nFirst layer weights shape:", self.first_layer_weights.shape)
        except AttributeError:
            raise AttributeError("Could not find 'conv1' layer in the model.")
    
    def visualize_filters(self, figsize: tuple = (12, 10)) -> None:
        """
        Visualize the filters in a grid layout.
        
        Args:
            figsize (tuple): Figure size in inches (width, height)
        """
        if self.first_layer_weights is None:
            raise ValueError("Weights not extracted. Call extract_first_layer_weights() first.")
            
        plt.figure(figsize=figsize)
        
        num_filters = self.first_layer_weights.shape[0]
        for i in range(num_filters):
            plt.subplot(3, 4, i + 1)
            filter_weights = self.first_layer_weights[i, 0].detach().numpy()
            plt.imshow(filter_weights, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.title(f'Filter {i+1}')
            
        plt.tight_layout()
        plt.show()
    
    def print_filter_weights(self) -> None:
        """Print the numerical values of each filter."""
        if self.first_layer_weights is None:
            raise ValueError("Weights not extracted. Call extract_first_layer_weights() first.")
            
        print("\nFilter weights:")
        for i in range(self.first_layer_weights.shape[0]):
            print(f"\nFilter {i+1}:")
            print(self.first_layer_weights[i, 0].detach().numpy())
    
    def visualize_filters_and_outputs(self, figsize: tuple = (20, 15)) -> None:
        """
        Visualize filters and their outputs in two columns of 5 filters each.
        
        Args:
            figsize (tuple): Figure size in inches (width, height)
        """
        if self.first_layer_weights is None:
            raise ValueError("Weights not extracted. Call extract_first_layer_weights() first.")
            
        # Get the first training example
        train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                                transform=torchvision.transforms.ToTensor())
        first_image = train_dataset[0][0].squeeze().numpy()
        
        # Create a figure to display filters and their outputs
        fig = plt.figure(figsize=figsize)
        
        # Create two subfigures (left and right columns)
        subfigs = fig.subfigures(1, 2, width_ratios=[1, 1])
        
        # Left column (first 5 filters)
        axs_left = subfigs[0].subplots(6, 2)  # 6 rows (original + 5 filters), 2 cols
        subfigs[0].suptitle('Filters 1-5', fontsize=16)
        
        # Right column (last 5 filters)
        axs_right = subfigs[1].subplots(6, 2)  # 6 rows (original + 5 filters), 2 cols
        subfigs[1].suptitle('Filters 6-10', fontsize=16)
        
        # Display original image in both columns
        axs_left[0, 0].imshow(first_image, cmap='gray')
        axs_left[0, 0].set_title('Original Image')
        axs_left[0, 0].axis('off')
        axs_left[0, 1].axis('off')  # Empty subplot for symmetry
        
        axs_right[0, 0].imshow(first_image, cmap='gray')
        axs_right[0, 0].set_title('Original Image')
        axs_right[0, 1].axis('off')  # Empty subplot for symmetry
        axs_right[0, 0].axis('off')
        
        with torch.no_grad():
            # Process first 5 filters (left column)
            for i in range(5):
                # Get and display the filter
                filter_weights = self.first_layer_weights[i, 0].numpy()
                axs_left[i+1, 0].imshow(filter_weights, cmap='gray')
                axs_left[i+1, 0].set_title(f'Filter {i+1}')
                axs_left[i+1, 0].axis('off')
                
                # Apply and display the filtered image
                filtered_image = cv2.filter2D(first_image, -1, filter_weights)
                axs_left[i+1, 1].imshow(filtered_image, cmap='gray')
                axs_left[i+1, 1].set_title(f'Output {i+1}')
                axs_left[i+1, 1].axis('off')
            
            # Process last 5 filters (right column)
            for i in range(5):
                # Get and display the filter
                filter_weights = self.first_layer_weights[i+5, 0].numpy()
                axs_right[i+1, 0].imshow(filter_weights, cmap='gray')
                axs_right[i+1, 0].set_title(f'Filter {i+6}')
                axs_right[i+1, 0].axis('off')
                
                # Apply and display the filtered image
                filtered_image = cv2.filter2D(first_image, -1, filter_weights)
                axs_right[i+1, 1].imshow(filtered_image, cmap='gray')
                axs_right[i+1, 1].set_title(f'Output {i+6}')
                axs_right[i+1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_network(self) -> None:
        """Perform complete analysis of the network."""
        self.load_model()
        self.extract_first_layer_weights()
        self.visualize_filters()
        self.print_filter_weights()
        self.visualize_filters_and_outputs()


def main():
    """Main function to run the network analysis."""
    try:
        # Create analyzer instance
        analyzer = NetworkAnalyzer('mnist_model.pth')
        
        # Perform analysis
        analyzer.analyze_network()
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    main()