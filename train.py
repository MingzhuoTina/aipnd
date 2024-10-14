import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
import argparse
import os


# Helper function to load data
def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    }

    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    }

    return dataloaders, image_datasets


# Function to build and train model
def build_and_train(data_dir, arch='vgg16', hidden_units=512, learning_rate=0.001, epochs=5, gpu=False, save_dir=''):
    dataloaders, image_datasets = load_data(data_dir)

    # Choose architecture
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        print(f"Architecture {arch} is not supported. Choose between 'vgg16' and 'vgg13'.")
        return

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define classifier
    input_size = model.classifier[0].in_features
    output_size = len(image_datasets['train'].classes)

    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # Training
    steps = 0
    print_every = 40
    running_loss = 0

    for epoch in range(epochs):
        model.train()

        for inputs, labels in dataloaders['train']:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in dataloaders['valid']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch + 1}/{epochs}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Validation loss: {valid_loss / len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy / len(dataloaders['valid']):.3f}")

                running_loss = 0
                model.train()

    # Save checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx

    checkpoint = {
        'arch': arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs
    }

    save_path = os.path.join(save_dir, 'checkpoint.pth') if save_dir else 'checkpoint.pth'
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset of images.")
    parser.add_argument('data_dir', help="Directory containing the dataset.")
    parser.add_argument('--arch', type=str, default='vgg16', help="Model architecture, vgg16 or vgg13 (default: vgg16)")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument('--hidden_units', type=int, default=512,
                        help="Number of hidden units in the classifier (default: 512)")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for training if available")
    parser.add_argument('--save_dir', type=str, default='', help="Directory to save checkpoints")

    args = parser.parse_args()

    build_and_train(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu,
                    args.save_dir)


if __name__ == "__main__":
    main()
