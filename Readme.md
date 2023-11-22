# Preparing Dataset

See FFCV_README.md from [original FFCV repo](https://github.com/libffcv/ffcv-imagenet) (also inclued here for completeness) for preparing imagenet data.

We use same parameters for preparing the dataset as the example using this command:

```bash
write_imagenet.sh 500 0.50 90
```

# Training Block Local Network

BLL imagenet can be trained using the following command:

```bash
python train_imagenet_BLL.py --config-file rn50_BLL_configs/rn50_32_epochs.yaml --training.epochs=150 --data.train_dataset=path/to/train/data --data.val_dataset=path/to/val/data --data.test_dataset=path/to/val/data

```

To train pipelined BLL ( blocks on multiple devices) we use the same parameters as above with a pipeline model

```bash
python train_imagenet_pipelinedBLL.py --config-file rn50_BLL_configs/rn50_32_epochs.yaml --model.arch=ResNet50BlocksPipeline --training.epochs=150 --data.train_dataset=path/to/train/data --data.val_dataset=path/to/val/data --data.test_dataset=path/to/val/data
```

# File description

Description of files included in this repo. Files marked with * are unchanged from original [FFCV](https://github.com/libffcv/ffcv-imagenet) repository.

``` 
.
├──* FFCV_README.md - Readme from original FFCV repo, included here for completeness
├── loss.py - File containing losses used to train BLL network
├── models.py - Block local models, see comments for description of each model
├── Readme.md - This file
├── requirements.txt - python requirements
├──* rn50_32_epochs.yaml - configuration file for Resnet50 trained for 32 epochs, from FFCV repo
├── rn50_BLL_32_epochs.yaml - configuration file for Resnet50 BLL model trained for 32 epochs
├── train_imagenet_BLL.py - Block local train script
├── train_imagenet_pipelinedBLL.py - Block local train script with pipelined implementation
├──* train_imagenet.py - original resnet trainer
├──* write_imagenet.py - original imagenet dataset creator
└──* write_imagenet.sh - helper script to create FFCV dataset
```