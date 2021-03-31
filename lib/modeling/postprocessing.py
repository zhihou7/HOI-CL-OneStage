# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch.nn import functional as F

from detectron2.structures import Instances


def detector_postprocess(results, output_height, output_width, mask_threshold=0.5):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    scale_x, scale_y = (output_width / results.image_size[1], output_height / results.image_size[0])
    results = Instances((output_height, output_width), **results.get_fields())

    for key in ["pred_boxes", "proposal_boxes", "person_boxes", "object_boxes"]:
        if results.has(key):
            output_boxes = results.get(key)

            output_boxes.scale(scale_x, scale_y)
            output_boxes.clip(results.image_size)

            results = results[output_boxes.nonempty()]

    return results
