import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchviz import make_dot
import sys


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 10 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        
        # Second convolutional layer: 10 input channels, 20 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        
        # Dropout layer with 0.25 dropout rate
        self.dropout = nn.Dropout2d(p=0.25)
        
        # First fully connected layer: 320 inputs (20 channels * 4 * 4), 50 outputs
        self.fc1 = nn.Linear(320, 50)
        
        # Output layer: 50 inputs, 10 outputs (one for each digit)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        # First conv layer followed by max pooling and ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        
        # Second conv layer with dropout, followed by max pooling and ReLU
        x = F.relu(F.max_pool2d(self.dropout(self.conv2(x)), 2))
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 320)
        
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer with log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

def train_network(model, train_loader, optimizer, epoch, device):
    """
    Train the network for one epoch
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        epoch: Current epoch number
        device: Device to run the training on (cuda/cpu)
    
    Returns:
        float: Average loss for this epoch
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to GPU
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += len(data)
        
    return total_loss / len(train_loader), 100. * correct / total

def test_network(model, test_loader, device):
    """
    Test the network on the test set
    
    Args:
        model: Neural network model
        test_loader: DataLoader for test data
        device: Device to run the testing on (cuda/cpu)
    
    Returns:
        tuple: (test accuracy, average loss)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to GPU
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(data)
    
    test_loss /= total
    accuracy = 100. * correct / total
    return accuracy, test_loss

def main(argv):
    """
    Main function to create, train and save the neural network
    
    Args:
        argv: Command line arguments
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
    
    test_dataset = torchvision.datasets.MNIST('./data', train=False,
                                            transform=torchvision.transforms.Compose([
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize the model, move it to GPU, and create optimizer
    model = NeuralNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    
    # Lists to store metrics for plotting
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    # Train for 10 epochs
    for epoch in range(1, 10):
        train_loss, train_acc = train_network(model, train_loader, optimizer, epoch, device)
        test_acc, test_loss = test_network(model, test_loader, device)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot training and testing accuracy
    ax1.plot(range(1, 10), train_accuracies, 'b-', label='Training Accuracy')
    ax1.plot(range(1, 10), test_accuracies, 'r-', label='Testing Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training and Testing Accuracy vs. Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and testing error (loss)
    ax2.plot(range(1, 10), train_losses, 'b-', label='Training Error')
    ax2.plot(range(1, 10), test_losses, 'r-', label='Testing Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Training and Testing Error vs. Epoch')
    ax2.legend()
    ax2.grid(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    
    # Save the trained model (save GPU model to CPU)
    torch.save(model.cpu().state_dict(), 'mnist_model.pth')

if __name__ == "__main__":
    main(sys.argv)
        