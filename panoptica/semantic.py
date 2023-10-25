from __future__ import annotations
from typing import Tuple
from multiprocessing import Pool

import numpy as np

import cc3d
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from .timing import measure_time


from .evaluator import Evaluator
from .result import PanopticaResult


class SemanticSegmentationEvaluator(Evaluator):
    def __init__(self, cca_backend: str):
        self.cca_backend = cca_backend

    @measure_time
    def evaluate(
        self,
        reference_mask: np.ndarray,
        prediction_mask: np.ndarray,
        iou_threshold: float,
    ):
        ref_labels, num_ref_instances = self._label_instances(
            mask=reference_mask,
        )

        pred_labels, num_pred_instances = self._label_instances(
            mask=prediction_mask,
        )

        self._handle_edge_cases(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
        )

        # Create a pool of worker processes to parallelize the computation

        with Pool() as pool:
            # Generate all possible pairs of instance indices for IoU computation
            instance_pairs = [
                (ref_labels, pred_labels, ref_idx, pred_idx)
                for ref_idx in range(1, num_ref_instances + 1)
                for pred_idx in range(1, num_pred_instances + 1)
            ]

            # Calculate IoU for all instance pairs in parallel using starmap
            iou_values = pool.starmap(self._compute_instance_iou, instance_pairs)

        # Reshape the resulting IoU values into a matrix
        iou_matrix = np.array(iou_values).reshape(
            (num_ref_instances, num_pred_instances)
        )

        # Use linear_sum_assignment to find the best matches
        ref_indices, pred_indices = linear_sum_assignment(-iou_matrix)

        # Initialize variables for True Positives (tp) and False Positives (fp)
        tp, fp, dice_list, iou_list = 0, 0, [], []

        # Loop through matched instances to compute PQ components
        for ref_idx, pred_idx in zip(ref_indices, pred_indices):
            iou = iou_matrix[ref_idx][pred_idx]
            if iou >= iou_threshold:
                # Match found, increment true positive count and collect IoU and Dice values
                tp += 1
                iou_list.append(iou)

                # Compute Dice for matched instances
                dice = self._compute_instance_volumetric_dice(
                    ref_labels=ref_labels,
                    pred_labels=pred_labels,
                    ref_instance_idx=ref_idx + 1,
                    pred_instance_idx=pred_idx + 1,
                )
                dice_list.append(dice)
            else:
                # No match found, increment false positive count
                fp += 1

        # Create and return the PanopticaResult object with computed metrics
        return PanopticaResult(
            num_ref_instances=num_ref_instances,
            num_pred_instances=num_pred_instances,
            tp=tp,
            fp=fp,
            dice_list=dice_list,
            iou_list=iou_list,
        )

    def _label_instances(
        self,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, int]:
        """
        Label connected components in a segmentation mask.

        Args:
            mask (np.ndarray): segmentation mask (2D or 3D array).
            cca_backend (str): Backend for connected components labeling. Should be "cc3d" or "scipy".

        Returns:
            Tuple[np.ndarray, int]:
                - Labeled mask with instances
                - Number of instances found
        """
        if self.cca_backend == "cc3d":
            labeled, num_instances = cc3d.connected_components(mask, return_N=True)
        elif self.cca_backend == "scipy":
            labeled, num_instances = ndimage.label(mask)
        else:
            raise NotImplementedError(f"Unsupported cca_backend: {self.cca_backend}")
        return labeled, num_instances

    def _compute_instance_iou(
        self,
        ref_labels: np.ndarray,
        pred_labels: np.ndarray,
        ref_instance_idx: int,
        pred_instance_idx: int,
    ) -> float:
        """
        Compute Intersection over Union (IoU) between a specific pair of reference and prediction instances.

        Args:
            ref_labels (np.ndarray): Reference instance labels.
            pred_labels (np.ndarray): Prediction instance labels.
            ref_instance_idx (int): Index of the reference instance.
            pred_instance_idx (int): Index of the prediction instance.

        Returns:
            float: IoU between the specified instances.
        """
        ref_instance_mask = ref_labels == ref_instance_idx
        pred_instance_mask = pred_labels == pred_instance_idx

        return self._compute_iou(
            reference=ref_instance_mask,
            prediction=pred_instance_mask,
        )

    def _compute_instance_dice_coefficient(
        self,
        ref_labels: np.ndarray,
        pred_labels: np.ndarray,
        ref_instance_idx: int,
        pred_instance_idx: int,
    ) -> float:
        """
        Compute the Dice coefficient between a specific pair of instances.

        The Dice coefficient measures the similarity or overlap between two binary masks representing instances.
        It is defined as:

        Dice = (2 * intersection) / (ref_area + pred_area)

        Args:
            ref_labels (np.ndarray): Reference instance labels.
            pred_labels (np.ndarray): Prediction instance labels.
            ref_instance_idx (int): Index of the reference instance.
            pred_instance_idx (int): Index of the prediction instance.

        Returns:
            float: Dice coefficient between the specified instances. A value between 0 and 1, where higher values
            indicate better overlap and similarity between instances.
        """
        ref_instance_mask = ref_labels == ref_instance_idx
        pred_instance_mask = pred_labels == pred_instance_idx

        return self._compute_dice_coefficient(
            reference=ref_instance_mask,
            prediction=pred_instance_mask,
        )