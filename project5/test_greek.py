import torch
import torchvision
from PIL import Image
from greek_classifier import GreekClassifier, GreekTransform

def test_image(model, image_path):
    """Test a single image."""
    # Load and transform the image
    image = Image.open(image_path)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        GreekTransform(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Process the image
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = output.max(1, keepdim=True)[1].item()
        
    # Map prediction to Greek letter
    letters = ['alpha', 'beta', 'gamma']
    return letters[pred]

def main():
    # Load the trained model
    model = GreekClassifier('mnist_model.pth')
    model.load_state_dict(torch.load('greek_model.pth'))
    
    # Test your images - update these paths to match your actual image locations
    test_images = [
        './greek_letters/alpha/my_alpha.png',  # Update with actual path
        './greek_letters/beta/my_beta.png',    # Update with actual path
        './greek_letters/gamma/my_gamma.png'   # Update with actual path
    ]
    
    for image_path in test_images:
        try:
            prediction = test_image(model, image_path)
            print(f'{image_path}: Predicted as {prediction}')
        except FileNotFoundError:
            print(f"Could not find image file: {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    main()