# Automated Treefall Detection Program

## Installation
From the [Releases page](https://github.com/matterport/Mask_RCNN/releases) page:
1. Download `mask_rcnn_treefall.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).


## Run Jupyter notebooks
Open the `inspect_treefall_data.ipynb` or `inspect_treefall_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Treefall model

Train a new model starting from pre-trained COCO weights
```
python3 treefall.py train --dataset=../../datasets/treefall/train --weights=coco
```

Resume training a model that you had trained earlier
```
python3 treefall.py train --dataset=../../datasets/treefall/train --weights=last
```

Train a new model starting from ImageNet weights
```
python3 treefall.py train --dataset=../../datasets/treefall/train --weights=imagenet
```

The code in `treefall.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.
