---
layout: post
title: FLIR ADAS Object Detection
description: PyTorch Object Detection Using The FLIR ADAS Image Set
image: assets/images/super_resolution_only.jpg
nav-menu: true
---

<!--
![Super Resolution Example](assets/images/super_resolution_only.jpg)
-->
The FLIR ADAS set contains...

Import the required packages:
```python
import pathlib
import torch
import torchvision
import torch.utils.data
from torchvision import models, datasets, transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
```
Load the data and ensure that cuda is using my GPU rather than the CPU
```python
IMAGES_PATH = pathlib.Path(r"C:\Users\Brenon\Desktop\FLIR_Thermal_Dataset\FLIR_ADAS_v2\images_thermal_train")
ANNOTATIONS_PATH = IMAGES_PATH / "coco_train.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

```python
class MyTransform:
    def __init__(self):
        self.transforms = transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
        ])

    def __call__(self, img, target):
        img = self.transforms(img)
        return img, target

dataset = CocoDetection(str(IMAGES_PATH), str(ANNOTATIONS_PATH), transforms=MyTransform())
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    collate_fn=lambda batch: tuple(zip(*batch)),
)
```
Display three random images from the data, one of the three is shown below
```python
def display_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0))
    for box in boxes:
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()

for _ in range(3):
    index = random.randint(0, len(dataset) - 1)
    image, target = dataset[index]
    display_image_with_boxes(image, target['boxes'])
```
![Super Resolution Example](assets/images/FLIR_ADAS_sample.png)

This code loads a Faster R-CNN model with ResNet 50 weights, creates an optimizer and scheduler, then does some error handling for batches that contain images with no bounding boxes. There is for sure a more elegant way of doing this, but this was a quick and temporary work around. 
```python
# Using Resnet weights
weights = models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

# Initialize the model with the specified pretrained weights and move to GPU
model = models.detection.fasterrcnn_resnet50_fpn(weights=weights).to(device)
model.train()

# Initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

# Training loop with error handling and detailed logging
for batch_idx, (imgs, targets) in enumerate(data_loader):
    try:
        print(f"Processing batch {batch_idx + 1}/{len(data_loader)}")

        # Move data to the GPU
        imgs = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass to calculate losses
        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimize
        losses.backward()
        optimizer.step()
        scheduler.step()  # Update the learning rate

        # Log image shapes and target types
        print(f"Image shapes: {[img.shape for img in imgs]}, Target types: {[type(target) for target in targets]}")

        # Log losses for each loss type in the model
        for name, loss_val in loss_dict.items():
            print(f"{name:<20}: {loss_val:.3f}")

    except Exception as e:
        # Log the error message and batch index
        print(f"Error processing batch {batch_idx + 1}: {str(e)}")
        continue  # Skip to the next batch
```
Save the model to use for inference later
```python
# Define a path for saving the entire model
COMPLETE_MODEL_SAVE_PATH = pathlib.Path(r"C:\Users\Brenon\Desktop\complete_model.pth")

# Save the entire model, optimizer, and any additional information
torch.save({
    'model': model,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict()
}, COMPLETE_MODEL_SAVE_PATH)

print(f"Complete model, optimizer, and scheduler saved at {COMPLETE_MODEL_SAVE_PATH}")

####
#below is to save state dictionary and optimizer settings


# Define a path for saving the model's state dictionary
MODEL_STATE_DICT_PATH = pathlib.Path(r"C:\Users\Brenon\Desktop\model_state_dict.pth")

# Save the model's state dictionary and optimizer's state dictionary
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, MODEL_STATE_DICT_PATH)

print(f"Model state dictionary and optimizer state saved at {MODEL_STATE_DICT_PATH}")
```









