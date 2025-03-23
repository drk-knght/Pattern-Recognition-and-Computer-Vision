import torch
import torchvision
import matplotlib.pyplot as plt
from neural_net import NeuralNet
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

class GreekClassifier(nn.Module):
    def __init__(self, base_model_path: str):
        super(GreekClassifier, self).__init__()
        
        # Load the base MNIST model
        self.base_model = NeuralNet()
        self.base_model.load_state_dict(torch.load(base_model_path))
        
        # Freeze all parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Store all layers except the last one
        self.features = nn.Sequential(
            self.base_model.conv1,
            self.base_model.conv2,
            self.base_model.dropout,
            self.base_model.fc1
        )
        
        # Replace the last layer with a new one for 3 classes
        self.fc2 = nn.Linear(50, 3)  # 50 inputs (from fc1), 3 outputs (alpha, beta, gamma)
    
    def forward(self, x):
        # Use the frozen features
        x = self.base_model.conv1(x)
        x = F.relu(F.max_pool2d(x, 2))
        x = self.base_model.conv2(x)
        x = F.relu(F.max_pool2d(self.base_model.dropout(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.base_model.fc1(x))
        
        # Use the new classification layer
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def train_epoch(model, device, train_loader, optimizer, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
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
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, avg_loss

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loader for Greek letters
    training_set_path = './greek_letters'  # Adjust path as needed
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=5,
        shuffle=True
    )
    
    # Create and configure the model
    model = GreekClassifier('mnist_model.pth').to(device)
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    
    # Lists to store metrics
    accuracies = []
    losses = []
    
    # Train for 50 epochs
    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        accuracy, loss = train_epoch(model, device, greek_train, optimizer, epoch)
        accuracies.append(accuracy)
        losses.append(loss)
    
    # Plot training metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig('greek_training_metrics.png')
    
    # Save the trained model
    torch.save(model.state_dict(), 'greek_model.pth')

if __name__ == "__main__":
    main()