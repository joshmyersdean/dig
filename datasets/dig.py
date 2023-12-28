import os
import json
import random
import pickle
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.data_utils import get_gt_indices, get_annotations_and_gt, ModelessSample


class DIGDataset(Dataset):
    """
    Dataset class for the DIG dataset.
    """

    def __init__(
        self,
        directory: str,
        prior_probability: float = 0.6,
        gesture_types: List[int] = [0, 1, 2, 3, 4],
        split: str = "train",
    ):
        """
        Initialize the DIGDataset instance.

        Parameters:
        directory (str): The path to the dataset directory.
        prior_probability (float): The probability of using prior data. Default is 0.6.
        gesture_types (list[int]): List of gesture types to be included. Default is [0, 1, 2, 3, 4].
        split (str): The dataset split type ('train', 'test', etc.). Default is "train".
        """
        assert 0 <= prior_probability <= 1, "Prior probability must be in [0,1]"
        assert 1 <= len(gesture_types) <= 5, "Support provided for 1-5 gesture types"

        self.directory = directory
        self.filenames = os.listdir(directory)
        self.prior_probability = prior_probability
        self.gesture_types = gesture_types
        self.split = split

        self.index_map = self._create_index_map()

    def _create_index_map(self) -> List[Tuple[str, int]]:
        """
        Create or load an index map for the dataset.

        Returns:
        list[tuple]: A list of tuples containing filename and index pairs.
        """
        save_path = f"{self.split}_imap.pkl"
        if os.path.isfile(save_path):
            with open(save_path, "rb") as f:
                index_map = pickle.load(f)
        else:
            index_map = [
                (filename, i)
                for filename in tqdm(self.filenames)
                for i in range(
                    len(json.load(open(os.path.join(self.directory, filename))))
                )
            ]
            with open(save_path, "wb") as f:
                pickle.dump(index_map, f)
        return index_map

    def __len__(self):
        """
        Return the number of items in the dataset.

        Returns:
        int: The length of the index map.
        """
        return len(self.index_map)

    def __getitem__(self, idx: int) -> ModelessSample:
        """
        Get a data item by index.

        Parameters:
        idx (int): The index of the data item.

        Returns:
        ModelessSample: A ModelessSample instance for the specified index.
        """
        filename, local_idx = self.index_map[idx]
        with open(os.path.join(self.directory, filename), "r") as file:
            data = json.load(file)

        sample = data[local_idx]
        use_prior = np.random.binomial(1, p=self.prior_probability)
        gesture = random.choice(self.gesture_types)
        obj_id = np.random.choice(list(set(sample["obj_id"])))
        valid_indices = get_gt_indices(sample, obj_id, gesture, bool(use_prior))
        data_list = get_annotations_and_gt(sample, valid_indices, obj_id)
        return random.choice(data_list)
