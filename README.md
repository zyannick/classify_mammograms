# Classify Mammography

This is an implementation of the model used to classify mammography between different view (CC et MLO). The default model is Resnet11 which provides the advantage to not under-fit as Alexnet and not over-fit as Resnet18

## Prerequisites

Mainly:

- Anaconda
- Tensorflow 1.13
- OpenCV

To run the program, you should install all the libraries in the file tf1.yml as follows:

```bash
conda env create -f tf1.yml
conda activate tf1
# you can run the code
```

## How to run

When you launch the code, the dataset is partitionned between test and train through the file 'data_helpers.py'. 

### Train

By default, you'll run Resnet11 model with 20 epochs

```bash
 python3 train.py  -num_epochs=20 -phase='train'
```

You can also choose another model such as Alexnet (alexnet), the learning rate, the step size (decay of learning rate) by:

```bash
 python3 train.py  -backbone='alexnet'  -init_lr=0.0001 -step_size=5 -nums_epochs=20 -phase='train'
```

### Testing

To test your model, you can specify the backbone

```bash
 python3 train.py  -backbone='resnet'  phase='test'
```

### Extract features and visualize (latent space)

The results will be placed in the model directory trough checkpoint path.

```bash
python3 train.py  -backbone='resnet'  phase='extract_features'
```

The features will be extracted and save in the model dir and a t-SNE + PCA will be applied to visualize the features and the images.

### Saliency maps

The results will be displayed in the model directory.

```bash
python3 train.py  -backbone='resnet'  phase='saliency_maps'
```

