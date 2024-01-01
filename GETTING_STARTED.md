# Getting Started
To get started with the DIG dataset, first download the COCO+LVIS dataset according to [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation). Then install the requirements with `pip install -r requirements.txt`.

## DIG Splits
Given the size of the dataset, we chunk the data into json files containing up to 100 samples each. Each sample is corresponds to an image ID with annotations for: multiple gesture types (e.g., lasso, click), previous segmentations, ground truths, and other optional metadata. For the test set, we also include annotations for the setting of selecting multiple disconnected regions. Once COCO+LVIS is downloaded, download the dig splits that are located in this [OneDrive Link](https://o365coloradoedu-my.sharepoint.com/:f:/g/personal/jomy5901_colorado_edu/EuRcuFEdTdhFje_uBObq-gYBx8y8xEOqMxi2BmoIRlYqiQ?e=rQ3TdP). 
When unzipped, they will create directories containing the chunked JSON files for all samples. If you are using our [dataloading implementation](dig/dataset.py), the unzipped directory will be the `json_directory`. The `img_directory` will correspond to where you downloaded the COCO+LVIS images.
More specifics about the structure of the annotations can be found in [DATA.md](DATA.md).

### Dataloading Implementation
 If you are using our [dataloading implementation](dig/dataset.py), to avoid loading in large JSON files, we precomputed an index map of all json files for efficient dataloading. To cut down on time to construct these index maps, we provide pre-computed, pickled, index maps in `dig/index_maps`.
 
## RICE
We provide a simple implementation of our proposed RICE evaluation metric in `dig/rice.py`. 
