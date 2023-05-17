# face-hair-segmentation
Detect face skin and hair

## Dataset

Training is done in some labeled part of the LFW dataset.

Download input images from [this link](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) and labels from [this link](http://vis-www.cs.umass.edu/lfw/part_labels/parts_lfw_funneled_gt_images.tgz).

Also read [this page](http://vis-www.cs.umass.edu/lfw/part_labels/) for license and more info.

## Model

The problem is to detect face and hair part of the face in a rough way. The accuracy and exact boundary is not important at all. So the precision will be sacrificed in favor of model simplicity and speed.

UNet has been used as network and cross entropy and focal loss are used to compare imbalance and focusing effects.

## Train
