import torch
from torchvision import models
import argparse
from PIL import Image
import numpy as np
import json


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.__dict__[checkpoint['arch']](pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((256, 256))

    left = (img.width - 224) / 2
    top = (img.height - 224) / 2
    right = (img.width + 224) / 2
    bottom = (img.height + 224) / 2
    img = img.crop((left, top, right, bottom))

    np_img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_img = (np_img - mean) / std
    np_img = np_img.transpose((2, 0, 1))

    return torch.tensor(np_img).float()


def predict(image_path, model, top_k=5, gpu=False):
    model.eval()
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    img = process_image(image_path).unsqueeze_(0).to(device)

    with torch.no_grad():
        logps = model.forward(img)

    ps = torch.exp(logps)
    top_p, top_class = ps.topk(top_k, dim=1)

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_class = [idx_to_class[c] for c in top_class.cpu().numpy()[0]]

    return top_p.cpu().numpy()[0], top_class


def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    parser.add_argument('image_path', help="Path to the input image.")
    parser.add_argument('checkpoint', help="Path to the model checkpoint.")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes (default: 5)")
    parser.add_argument('--category_names', type=str, default='',
                        help="Path to a JSON file mapping categories to real names")
    parser.add_argument('--gpu', action='store_true', help="Use GPU for inference if available")

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)

    probs, classes = predict(args.image_path, model, args.top_k, args.gpu)

    # Map categories to real names if JSON file provided
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[c] for c in classes]

    for i in range(len(classes)):
        print(f"{classes[i]}: {probs[i]:.4f}")


if __name__ == "__main__":
    main()
