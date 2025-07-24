import cProfile

from pathlib import Path
import numpy as np
from panoptica.utils.input_check_and_conversion.check_nibabel_image import load_nibabel_image

def read_nifti_as_numpy(path):
    nib_image = load_nibabel_image(path)
    return np.asanyarray(nib_image.dataobj, dtype=nib_image.dataobj.dtype).copy()

from panoptica import (
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
    Panoptica_Evaluator,
    InputType,
)

directory = str(Path(__file__).absolute().parent)

reference_mask = read_nifti_as_numpy(directory + "/spine_seg/semantic/ref.nii.gz")
prediction_mask = read_nifti_as_numpy(directory + "/spine_seg/semantic/pred.nii.gz")


evaluator = Panoptica_Evaluator(
    expected_input=InputType.SEMANTIC,
    instance_approximator=ConnectedComponentsInstanceApproximator(),
    instance_matcher=NaiveThresholdMatching(),
    verbose=True,
    log_times=True,
)


def main():
    with cProfile.Profile() as pr:
        result = evaluator.evaluate(prediction_mask, reference_mask)["ungrouped"]

        # To print the results, just call print
        print(result)

        intermediate_steps_data = result.intermediate_steps_data
        assert intermediate_steps_data is not None
        # To get the different intermediate arrays, just use the second returned object
        intermediate_steps_data.original_prediction_arr  # Input prediction array, untouched
        intermediate_steps_data.original_reference_arr  # Input reference array, untouched

        intermediate_steps_data.prediction_arr(
            InputType.MATCHED_INSTANCE
        )  # Prediction array after instances have been matched
        intermediate_steps_data.reference_arr(
            InputType.MATCHED_INSTANCE
        )  # Reference array after instances have been matched

    pr.dump_stats(directory + "/semantic_example.log")
    return result, intermediate_steps_data


if __name__ == "__main__":
    main()
