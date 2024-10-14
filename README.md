
# Flower Image Classifier

This project trains a deep neural network to classify images of flowers into their respective categories. The project consists of two main scripts: `train.py` and `predict.py`.

## Project Structure
- `train.py`: Used to train a new neural network on a dataset of images and save the model as a checkpoint.
- `predict.py`: Used to predict the class of an image using a trained model checkpoint.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- numpy
- Pillow
- argparse

## Instructions

### 1. Training a Model

To train a new model using the flower dataset:

```bash
python train.py data_directory --save_dir save_directory --arch vgg13 --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu
```

- `data_directory`: Directory where the flower dataset is located.
- `--save_dir`: Directory to save the trained model checkpoint.
- `--arch`: The model architecture to use (default: vgg16, or vgg13).
- `--learning_rate`: Learning rate for training (default: 0.001).
- `--hidden_units`: Number of hidden units in the classifier (default: 512).
- `--epochs`: Number of training epochs (default: 5).
- `--gpu`: Use GPU for training if available.

Example:

```bash
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

### 2. Predicting Flower Species

To predict the class of a flower image using a trained model:

```bash
python predict.py /path/to/image checkpoint --top_k 3 --category_names cat_to_name.json --gpu
```

- `/path/to/image`: Path to the flower image you want to classify.
- `checkpoint`: Path to the model checkpoint file.
- `--top_k`: Return the top K most likely classes (default: 5).
- `--category_names`: Path to a JSON file mapping category labels to actual names.
- `--gpu`: Use GPU for inference if available.

Example:

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
```

### 3. Data Preprocessing

The flower dataset is divided into training, validation, and test directories. The `train.py` script automatically applies data transformations like random cropping, scaling, rotation, flipping, and normalization for the training dataset. For validation and testing datasets, it applies resizing, center cropping, and normalization.

### 4. Saving and Loading Checkpoints

- The `train.py` script saves the trained model as a checkpoint, which can later be loaded by `predict.py`.
- The checkpoint contains the model architecture, the state dictionary, the class-to-index mapping, and the optimizer state.

### Example Workflows

1. **Training a model**:
```bash
python train.py flowers --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 5 --gpu
```

2. **Predicting with a trained model**:
```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

## Notes
- Make sure that the flower dataset is organized into `train`, `valid`, and `test` subdirectories, each containing images in their respective class folders.
- The scripts are designed to run on both CPU and GPU. Use the `--gpu` flag to take advantage of a GPU if available.
