import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import argparse

# Define your validation transformations
transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define validation dataset class (similar to CustomImageNetDataset)
class CustomImageNetValidationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_int = {}
        self.int_to_label = {}

        # Collect class labels
        class_idx = 0
        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path):
                self.label_to_int[label_dir] = class_idx
                self.int_to_label[class_idx] = label_dir
                class_idx += 1

        # Traverse the directory and collect image paths and labels
        for label_dir in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    self.image_paths.append(os.path.join(label_path, img_name))
                    self.labels.append(self.label_to_int[label_dir])  # Convert label to integer

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            
            image = Image.new('RGB', (224, 224), (0, 0, 0))  

        label = self.labels[idx] 

        if self.transform:
            image = self.transform(image)

        return image, label

# Create ensemble prediction function using averaging
def ensemble_predict(models_list, input_tensor):
    outputs = []
    with torch.no_grad():
        for model in models_list:
            model.eval()  # Set model to evaluation mode
            output = model(input_tensor)
            outputs.append(output)

    # Average the outputs
    averaged_output = torch.stack(outputs).mean(dim=0)
    _, predicted_class = torch.max(averaged_output, 1)
    return predicted_class.item()

# Validation loop with ensemble
def validate_ensemble(models_list, device, val_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Get predictions from the ensemble
            predicted_class = ensemble_predict(models_list, data)
            
            total += target.size(0)
            correct += (predicted_class == target).sum().item()

    accuracy = 100. * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

# Main function with validation
def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet ResNet50 Ensemble Validation Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for validation (default: 256)')
    parser.add_argument('--val-dir', type=str, default='./data', help='Directory for validation data')
    args = parser.parse_args()

    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Load the ensemble of 5 models
    models_list = []
    for i in range(5):  
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust final layer for 1000 ImageNet classes
        model.load_state_dict(torch.load(f'subset_model{i+1}.pt'))  # Load each model's weights
        model.to(device)
        models_list.append(model)

    # Validation dataset and loader
    val_dataset = CustomImageNetValidationDataset(root_dir='./data', transform=transform_val)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Validation with ensemble
    print('\nValidation with Ensemble:')
    validate_ensemble(models_list, device, val_loader)

if __name__ == '__main__':
    main()
