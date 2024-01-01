# DATA Information

## About
All annotations are based off COCO+LVIS and we provide annotations for: clicks, scribbles, loose lassos, tight lassos, and rectangles. Our final dataset consists of a: training
set with 99,161 images, 857,669 regions, 186,740 region parts, and 12,839,936 samples; validation set with 3,318 images, 32,938 regions, 5698 region parts, and 413,784 samples; and test set with 1,423 images, 11,703 regions, 2417
region parts, and 246,080 samples. Of the 246,080 samples in the test set, 69,852 samples belong to the scenario of multi-region segmentation.

## Supported Gestures
We provide support for 5 different gesture types indexed in the range [0,4].
```python
{
  0: 'click',
  1: 'scribble',
  2: 'loose lasso',
  3: 'tight lasso',
  4: 'rectangle'
}
```

## JSON Structure
While we provide code to simplify the loading of DIG data, we provide further metadata in the JSON files. All annotations/GTs/previous segmentations are stored in RLE format.

- `img_id`: Prefix of image/gt in COCO+LVIS. e.g., `IMG_ID.jpg/IMG_ID.png`.
- `size`: Dimensions of the sample.
- `mul_obj_id`: Object ID in original ground truth for multi-region selection setting. [TEST ONLY]
- `obj_id`: Object ID in original ground truth.
- `syn_gt`: Valid pixels for synthetic (local) GT.
- `syn_gt_void`: Void pixels for synthetic (local) GT. Value is 255.
- `syn_annotations`: Multi-gesture annotations.
- `has_prior`: Boolean value indicated that a sample at index `i` has a previous segmentation.
- `prior`: Previous segmentation.
- `corr_id`: Which correction the prior and annotation correspond to.
- `gesture`: What type of gesture. Ranges from [0,4].
- `mode`: If the gesture indicates content to be added or removed.
- `has_mul`: Indicates if a sample at index `i` has multi-region annotations. [TEST ONLY].
- `mul_gesture`: Gesture for multi-region setting. [TEST ONLY].
- `mul_gt`: GT for multi-region setting, no void pixels. [TEST ONLY].
- `mul_annot`: Gesture annotation for multi-region setting. [TEST ONLY].
- `mul_prior`: Previous segmentation for multi-region setting. [TEST ONLY].

