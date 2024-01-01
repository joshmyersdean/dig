# Interactive Segmentation for Diverse Gesture Types Without Context
![A diagram show our pipeline for gesture-agnostic, context-free interactive segmentation. The left column shows a user, a turtle, and an optional previously colored turtle. The middle column shows 4 turtles with 4 different interactive gestures. The last column shows a colored in turtle.](static/teaser.png)

__Josh Myers-Dean, Yifei Fan, Brian Price, Wilson Chan, Danna Gurari__

[Paper](https://openaccess.thecvf.com/content/WACV2024/papers/Myers-Dean_Interactive_Segmentation_for_Diverse_Gesture_Types_Without_Context_WACV_2024_paper.pdf) [Supplementals](https://openaccess.thecvf.com/content/WACV2024/supplemental/Myers-Dean_Interactive_Segmentation_for_WACV_2024_supplemental.pdf) [Download](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/jomy5901_colorado_edu/EuRcuFEdTdhFje_uBObq-gYBx8y8xEOqMxi2BmoIRlYqiQ)

## TODOS
- Release Test Set

## About
This repository provides links to download data for the DIG dataset. Setup instructions are provided in [GETTING_STARTED.md](GETTING_STARTED.md). Given the size of the dataset, we chunk the data into json files containing up to 100 samples each. Each sample is corresponds to an image ID with annotations for: multiple gesture types (e.g., lasso, click), previous segmentations, ground truths, and other optional metadata. For the test set, we also include annotations for the setting of selecting multiple disconnected regions. For more info on the JSON, refer to [DATA.md](DATA.md).

### Using DIG
We provide a lightweight example of dataloading with dig in `dig/dataset.py`. 
```python
from dig.dataset import DIGDataset
ds = DIGDataset(JSON_DIRECTORY, IMG_DIRECTORY, split=SPLIT)
sample = ds[0] # sample.img, sample.previous_seg, sample.gt, sample.annotation
```

We provoide an implementation of our RICE evaluation metric in `dig/rice.py`.

## Issues
If you encounter any issues with the data or have any questions we encourage you to submit an issue in this repository. For all other inquiries, please reach out to josh [dot] myers-dean [at] colorado [dot] edu.

## Citing
If you find our work useful, please considering citing.
```
@InProceedings{Myers-Dean_2024_WACV,
    author    = {Myers-Dean, Josh and Fan, Yifei and Price, Brian and Chan, Wilson and Gurari, Danna},
    title     = {Interactive Segmentation for Diverse Gesture Types Without Context},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {7198-7208}
}
```
