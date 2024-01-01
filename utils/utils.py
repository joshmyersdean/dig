import numpy as np
from pycocotools import mask


class ModelessSample:
    """
    Represents a sample without a mode, including annotations and ground truth data.
    """

    def __init__(
        self, annotation: np.ndarray, previous_seg: np.ndarray, gt: np.ndarray, img: np.ndarray = None
    ):
        """
        Initialize a ModelessSample instance.

        Parameters:
        annotation (np.array): The annotation data.
        previous_seg (np.array): The segmentation from the previous step.
        gt (np.array): The ground truth data.
        img (np.array): image for sample
        """
        self.annotation = annotation
        self.previous_seg = previous_seg
        self.gt = gt
        self.img = img


def get_gt_indices(
    data: dict, obj_id: int, gesture: int, use_prior: bool = False
) -> list[int]:
    """
    Get the ground truth indices for given object ID and gesture.

    Parameters:
    data (dict): The dataset containing object IDs, gestures, and prior information.
    obj_id (int): The object ID to match.
    gesture (int): The gesture type to match.
    use_prior (bool): Whether to use prior data (previous seg) in filtering. Default is False.

    Returns:
    list[int]: A list of indices where the conditions are met.
    """
    obj_ids = np.array(data["obj_id"])
    gestures = np.array(data["gesture"])
    has_prior = np.array(data["has_prior"])

    # Combine conditions: matching object ID and gesture, and has_prior status based on use_prior
    condition = (obj_ids == obj_id) & (gestures == gesture) & (has_prior == use_prior)
    return np.where(condition)[0].tolist()


def get_annotations_and_gt(
    data: dict, indices: list[int], obj_id: int
) -> list[ModelessSample]:
    """
    Get annotations and ground truth data for given indices and object ID.

    Parameters:
    data (dict): The dataset containing syn_gt, syn_gt_void, and syn_annotations.
    indices (list[int]): Indices of each list in the data dict.
    obj_id (int): Object ID in scene.

    Returns:
    list[ModelessSample]: A list of ModelessSample instances.
    """
    results = []

    for i in indices:
        # Decode the masks for valid and void ground truths
        syn_gt_valid = mask.decode(data["syn_gt"][i])
        syn_gt_void = mask.decode(data["syn_gt_void"][i])

        # Combine the valid and void ground truths
        syn_gt = syn_gt_valid + (syn_gt_void * 255)
        annot = mask.decode(data["syn_annotations"][i])

        # Initialize with prior if available, else with zeros
        init = (
            mask.decode(data["prior"][str(obj_id)])
            if data["has_prior"][i]
            else np.zeros_like(annot)
        )

        results.append(ModelessSample(annot, init, syn_gt))

    return results
