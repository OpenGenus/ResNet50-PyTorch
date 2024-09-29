# ResNet50-PyTorch
The ResNet50 model training and inference in PyTorch

## Project Overview
This project implements a foundational ResNet50 model using PyTorch. The model is trained on a custom dataset of healthy and diseased(rust, scab) apple leaves. This README provides instructions to set up, run, and evaluate the model.

## Prerequisites
The below dependencies are to be installed before running the training model. For the versions used, please refer the requirements.txt file.

- Python 3.7 or higher
- PyTorch
- torchvision
- Numpy

## Setup
1. Clone the repository using `git clone <URL>`
2. Check all the dependencies to be installed.
3. Train the model on a dataset : `python resnet50_model.py`
4. Model inference and evaluation : Once the model is saved, go ahead and experiment with the model by fine-tuning on new data or different datasets. 
The model in this repo `saved_model/resnet50_apple_leaf_disease.pth` can also be used.
Some examples are given below : 

i. Load the saved model : 
```python
# Initialize model
num_classes = <number_of_classes>
model = ResNet50(num_classes=num_classes)

# Load the saved state dictionary
model.load_state_dict(torch.load('resnet50_apple_leaf_disease.pth'))
model.eval()
print("Model loaded and set to evaluation mode.")
```

ii. Print the model architecture : 
```python
print(model)
for name, param in model.named_parameters():
    print(name, param.data)
```
iii. Check accuracy :

```python
accuracy = evaluate_model(model, test_loader)
print(f"Accuracy : {accuracy:.2f}%")
```
## Further Exploration
The current model has a total accuracy of 93.97%. This can be further optimized to meet particular project requirements using data augmentation, fine-tuning the model and other advanced optimizers.
