import torch
from torch import nn
from torch.nn import functional as F
from fvcore.nn import smooth_l1_loss

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat
from detectron2.modeling.roi_heads import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage
from detectron2.data.catalog import MetadataCatalog
from detectron2.structures import Boxes, Instances


from .zero_shot import load_semantic_embeddings, ZeroShotPredictor
import numpy as np
inst_nums = np.array([1.760e+02, 9.800e+01, 5.600e+01, 1.810e+02, 1.980e+02, 7.500e+01,
       2.840e+02, 3.190e+02, 4.000e+00, 2.740e+02, 2.300e+01, 1.995e+03,
       1.380e+02, 1.210e+02, 2.700e+01, 1.700e+01, 1.160e+02, 8.700e+01,
       2.205e+03, 1.904e+03, 2.250e+03, 1.800e+02, 2.000e+00, 3.840e+02,
       4.500e+01, 2.570e+02, 1.270e+02, 5.000e+00, 1.600e+01, 4.390e+02,
       8.500e+01, 1.030e+02, 2.140e+02, 3.900e+01, 1.170e+02, 1.100e+01,
       1.760e+02, 6.300e+01, 4.051e+03, 7.570e+02, 3.800e+02, 2.385e+03,
       1.625e+03, 4.400e+01, 2.000e+00, 4.010e+02, 1.540e+02, 1.600e+02,
       5.370e+02, 5.100e+01, 7.000e+00, 1.600e+01, 2.300e+01, 5.270e+02,
       2.470e+02, 6.000e+00, 4.630e+02, 2.900e+01, 2.700e+01, 7.100e+01,
       1.518e+03, 9.910e+02, 6.000e+00, 1.000e+00, 2.260e+02, 2.700e+01,
       1.000e+00, 3.670e+02, 6.000e+01, 9.400e+01, 1.000e+00, 2.200e+01,
       2.600e+01, 5.910e+02, 1.750e+02, 1.051e+03, 1.000e+00, 4.000e+00,
       1.580e+02, 7.600e+01, 6.000e+00, 7.000e+01, 4.900e+01, 3.000e+00,
       1.000e+00, 1.020e+02, 5.200e+01, 1.120e+02, 8.000e+01, 1.266e+03,
       1.000e+00, 3.270e+02, 2.800e+01, 2.280e+02, 9.080e+02, 5.500e+01,
       1.000e+01, 2.450e+02, 9.600e+01, 5.000e+00, 5.000e+00, 5.200e+01,
       3.200e+01, 4.900e+01, 6.000e+00, 1.260e+02, 4.920e+02, 5.000e+00,
       2.336e+03, 3.205e+03, 2.350e+02, 9.700e+01, 2.000e+00, 1.200e+01,
       1.100e+01, 3.260e+02, 2.600e+01, 1.760e+02, 1.800e+01, 2.000e+01,
       1.620e+02, 2.000e+01, 8.300e+01, 1.600e+01, 5.000e+01, 1.280e+02,
       2.500e+01, 3.000e+00, 2.110e+02, 2.300e+01, 3.600e+01, 8.290e+02,
       2.600e+01, 1.410e+02, 1.100e+01, 1.000e+00, 8.000e+00, 8.600e+01,
       2.250e+02, 1.525e+03, 5.100e+02, 1.190e+03, 8.900e+01, 1.000e+02,
       2.600e+01, 6.200e+01, 9.780e+02, 6.900e+01, 1.240e+02, 3.000e+00,
       1.060e+02, 1.200e+01, 4.470e+02, 1.365e+03, 1.686e+03, 1.590e+03,
       2.580e+02, 1.500e+01, 3.000e+00, 5.870e+02, 8.700e+01, 1.900e+01,
       5.180e+02, 2.620e+02, 6.200e+01, 5.000e+00, 1.000e+00, 1.500e+02,
       7.000e+00, 6.620e+02, 1.000e+01, 1.000e+01, 9.000e+00, 2.330e+02,
       2.100e+01, 9.300e+01, 5.710e+02, 1.750e+02, 3.200e+01, 2.000e+00,
       2.600e+01, 1.000e+00, 1.110e+02, 1.290e+02, 5.000e+00, 5.090e+02,
       1.110e+02, 1.180e+02, 4.000e+00, 2.000e+00, 5.820e+02, 3.050e+02,
       8.000e+00, 2.640e+02, 3.100e+01, 3.000e+00, 2.250e+02, 2.700e+02,
       7.000e+00, 1.800e+01, 5.000e+01, 2.230e+02, 2.900e+01, 2.500e+01,
       2.800e+01, 2.000e+00, 2.000e+00, 4.500e+01, 9.440e+02, 8.000e+01,
       2.000e+01, 1.600e+01, 1.021e+03, 1.470e+02, 9.000e+00, 3.270e+02,
       4.000e+00, 1.160e+02, 3.180e+02, 2.020e+02, 1.100e+01, 2.600e+01,
       3.000e+00, 3.420e+02, 5.000e+01, 8.300e+01, 7.190e+02, 1.000e+00,
       3.360e+02, 2.000e+00, 6.230e+02, 4.900e+01, 1.980e+02, 1.990e+02,
       1.100e+01, 2.100e+01, 6.100e+01, 1.000e+02, 3.000e+00, 2.000e+00,
       3.280e+02, 1.530e+02, 3.300e+01, 1.300e+01, 8.700e+01, 2.013e+03,
       2.330e+02, 1.270e+02, 8.230e+02, 5.620e+02, 6.960e+02, 2.150e+02,
       7.500e+01, 3.400e+01, 3.000e+00, 2.000e+00, 6.300e+01, 1.000e+00,
       4.500e+01, 4.500e+01, 1.000e+00, 2.000e+00, 1.000e+00, 2.600e+01,
       3.210e+02, 1.020e+02, 2.980e+02, 7.500e+01, 1.430e+02, 3.000e+01,
       1.100e+01, 7.600e+01, 1.780e+02, 1.010e+02, 4.000e+00, 2.600e+01,
       3.800e+01, 1.970e+02, 2.200e+01, 1.000e+00, 1.000e+00, 3.000e+00,
       4.700e+01, 1.830e+02, 5.380e+02, 1.880e+02, 1.000e+00, 2.350e+02,
       1.310e+02, 1.000e+00, 6.900e+01, 4.400e+01, 6.000e+00, 1.100e+01,
       4.760e+02, 1.390e+02, 1.150e+02, 3.120e+02, 1.500e+01, 1.500e+01,
       8.100e+01, 3.500e+01, 5.100e+01, 1.000e+00, 2.920e+02, 1.000e+01,
       5.800e+01, 1.230e+02, 2.730e+02, 4.400e+01, 8.300e+01, 1.000e+00,
       2.760e+02, 4.200e+01, 1.070e+02, 2.000e+00, 1.300e+01, 2.000e+00,
       1.400e+01, 7.900e+01, 4.610e+02, 1.930e+02, 5.400e+01, 6.630e+02,
       1.100e+01, 1.000e+00, 2.800e+01, 1.500e+01, 9.000e+00, 2.340e+02,
       1.400e+02, 1.110e+02, 2.900e+01, 6.000e+00, 2.000e+00, 1.310e+02,
       9.700e+01, 3.230e+02, 4.420e+02, 1.700e+01, 2.390e+02, 8.900e+01,
       1.630e+02, 1.200e+01, 2.300e+01, 2.000e+00, 4.110e+02, 1.040e+02,
       4.900e+01, 5.000e+01, 2.000e+00, 1.000e+00, 7.600e+01, 1.560e+02,
       6.000e+00, 3.700e+01, 1.460e+02, 1.160e+02, 1.000e+00, 1.050e+02,
       2.200e+02, 5.000e+01, 9.600e+01, 4.400e+01, 3.000e+00, 4.800e+01,
       2.180e+02, 1.040e+02, 4.300e+01, 2.190e+02, 7.240e+02, 5.390e+02,
       1.120e+02, 1.520e+02, 4.420e+02, 1.140e+02, 2.100e+02, 5.720e+02,
       4.000e+01, 1.000e+00, 5.820e+02, 5.000e+00, 3.900e+01, 2.280e+02,
       3.600e+01, 6.220e+02, 1.400e+01, 2.460e+02, 1.910e+02, 1.000e+00,
       5.000e+00, 7.000e+00, 1.800e+01, 4.700e+01, 5.600e+01, 1.000e+00,
       4.200e+01, 1.000e+00, 3.000e+00, 3.000e+00, 3.700e+01, 1.000e+00,
       1.000e+00, 3.000e+00, 7.000e+00, 2.000e+00, 1.280e+02, 4.000e+00,
       1.300e+01, 8.100e+01, 5.000e+00, 1.800e+01, 2.300e+01, 6.900e+01,
       3.600e+01, 2.400e+01, 1.000e+00, 2.710e+02, 1.000e+00, 8.300e+01,
       3.600e+01, 4.900e+01, 1.240e+02, 1.840e+02, 5.900e+01, 5.200e+01,
       5.000e+00, 1.000e+00, 1.120e+02, 2.000e+00, 3.800e+01, 4.000e+00,
       2.200e+01, 3.600e+01, 1.860e+02, 6.200e+01, 3.000e+00, 2.100e+01,
       3.000e+01, 1.000e+00, 3.000e+00, 1.200e+02, 1.970e+02, 1.200e+01,
       8.500e+01, 1.480e+02, 2.470e+02, 8.900e+01, 2.500e+01, 2.000e+00,
       1.200e+01, 9.000e+00, 7.900e+01, 1.370e+02, 4.350e+02, 2.580e+02,
       4.100e+02, 9.820e+02, 5.800e+01, 1.681e+03, 2.200e+01, 1.488e+03,
       1.230e+02, 5.000e+00, 6.600e+01, 9.000e+01, 2.100e+01, 1.240e+02,
       2.900e+01, 5.000e+00, 6.850e+02, 8.920e+02, 1.013e+03, 1.900e+01,
       5.000e+00, 3.800e+01, 6.100e+01, 1.570e+02, 3.550e+02, 6.310e+02,
       6.510e+02, 7.210e+02, 3.000e+00, 2.440e+02, 9.300e+01, 1.000e+00,
       3.200e+01, 1.340e+02, 6.000e+01, 1.790e+02, 9.100e+01, 1.300e+02,
       2.340e+02, 3.390e+02, 1.110e+02, 2.990e+02, 1.100e+01, 6.800e+01,
       1.000e+00, 1.000e+00, 1.410e+02, 2.200e+01, 2.200e+01, 1.000e+01,
       1.000e+00, 4.400e+01, 2.310e+02, 1.150e+02, 2.380e+02, 1.000e+00,
       1.000e+01, 1.100e+01, 3.000e+01, 3.000e+01, 1.000e+00, 7.800e+02,
       2.280e+02, 6.000e+00, 3.140e+02, 1.500e+01, 3.000e+00, 2.900e+01,
       2.000e+00, 2.210e+02, 1.540e+02, 2.500e+01, 1.000e+00, 1.800e+02,
       4.700e+01, 2.760e+02, 5.600e+01, 3.000e+00, 1.470e+02, 2.000e+02,
       2.430e+02, 1.000e+00, 8.500e+01, 3.300e+01, 5.700e+01, 1.000e+00,
       8.300e+01, 1.300e+01, 3.000e+01, 2.800e+01, 8.210e+02, 2.030e+02,
       2.000e+00, 2.000e+00, 1.000e+00, 2.000e+00, 7.000e+00, 1.000e+00,
       3.000e+00, 1.500e+01, 5.600e+01, 3.000e+00, 5.000e+00, 1.100e+02,
       2.320e+02, 3.200e+02, 1.000e+00, 1.900e+01, 1.300e+01, 1.700e+01,
       2.100e+01, 2.400e+01, 7.250e+02, 2.400e+01, 1.470e+02, 4.900e+01,
       1.100e+02, 2.000e+01, 2.780e+02, 1.970e+02, 1.400e+01, 2.470e+02,
       1.081e+03, 1.477e+03, 6.000e+00, 1.400e+01, 9.000e+00, 9.000e+00,
       1.271e+03, 5.290e+02, 1.500e+01, 1.700e+01, 1.000e+00, 1.810e+02,
       2.800e+01, 4.090e+02, 4.700e+01, 7.500e+01, 1.000e+00, 1.000e+00,
       4.220e+02, 8.000e+00, 2.000e+00, 4.000e+00, 7.600e+01, 2.000e+00])


def interaction_inference_single_image(
    image_shape,
    person_boxes,
    object_boxes,
    person_box_scores,
    object_box_scores,
    object_box_classes,
    hoi_scores,
    score_thresh,
    topk_per_image
):
    """
    Single-image HOI inference.
    Return HOI detection results by thresholding on scores.

    Args:
        image_shape (tuple): (width, height) tuple for each image in the batch.
        person_boxes (Boxes): A `Boxes` has shape (N, 4), where N is the number of person boxes.
        object_boxes (Boxes): A `Boxes` has shape (M, 4), where M is the number of object boxes.
        person_box_scores (Tensor): A Tensor of predicted pesron box scores with shape (N, ).
        object_box_scores (Tensor): A Tensor of predicted object box scores with shape (M, ).
        object_box_classes (Tensor): A Tensor of predicted object box classes with shape (M, 1).
        hoi_scores (Tensor): A Tensor has shape (N, M, K), where K is the number of actions.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.
    
    Returns:
        instances: (Instances): An `Instances` that stores the topk most confidence detections.
    """
    # Reweight interaction scores with (person & object) box scores
    box_scores = person_box_scores * object_box_scores
    scores = hoi_scores *  box_scores[:, None].repeat(1, hoi_scores.size(-1))

    # Filter results based on detection scores
    filter_mask = scores > score_thresh # (N, M, K)
    # (R, 2. First column contains indices coresponding to predictions.
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()

    person_boxes = person_boxes[filter_inds[:, 0]]
    object_boxes = object_boxes[filter_inds[:, 0]]
    object_classes = object_box_classes[filter_inds[:, 0]]
    action_classes = filter_inds[:, 1]
    scores = scores[filter_mask]

    if topk_per_image > 0 and topk_per_image < len(filter_inds):
        keep = torch.argsort(scores, descending=True)
        keep = keep[:topk_per_image]
        person_boxes, object_boxes = person_boxes[keep], object_boxes[keep]
        object_classes, action_classes = object_classes[keep], action_classes[keep]
        scores = scores[keep]

    result = Instances(image_shape)
    result.person_boxes = person_boxes
    result.object_boxes = object_boxes
    result.object_classes = object_classes
    result.action_classes = action_classes
    result.scores = scores
    return result


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
        self,
        box2box_transform,
        pred_class_logits,
        pred_proposal_deltas,
        proposals,
        smooth_l1_beta=0,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * F.cross_entropy(
                self.pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * smooth_l1_loss(
                self.pred_proposal_deltas,
                torch.zeros_like(self.pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)).squeeze(
            1
        )
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.view(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1).expand(num_pred, K, B).reshape(-1, B),
        )
        return boxes.view(num_pred, K * B)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.0
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_boxes_for_gt_classes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        predicted_boxes = self._predict_boxes()
        B = self.proposals.tensor.shape[1]
        # If the box head is class-agnostic, then the method is equivalent to `predicted_boxes`.
        if predicted_boxes.shape[1] > B:
            num_pred = len(self.proposals)
            num_classes = predicted_boxes.shape[1] // B
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = torch.clamp(self.gt_classes, 0, num_classes - 1)
            predicted_boxes = predicted_boxes.view(num_pred, num_classes, B)[
                torch.arange(num_pred, dtype=torch.long, device=predicted_boxes.device), gt_classes
            ]
        return predicted_boxes.split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class BoxOutputs(FastRCNNOutputs):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    It provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def box_inference(self, score_thresh, nms_thresh, topk_per_image):  
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.pred_class_logits.split(self.num_preds_per_image, dim=0)
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * F.cross_entropy(
                self.pred_class_logits,
                torch.zeros(0, dtype=torch.long, device=self.pred_class_logits.device),
                reduction="sum",
            )
        else:
            self._log_accuracy()
            # See ``:class:StandardHOROIHeads._forward_box``. Note that we have computed the
            # softmax of box scores at ``_reweight_box_given_proposal_scores``. Thus, here we
            # apply F.nll_loss() instead of F.cross_entropy()
            return F.nll_loss(torch.log(self.pred_class_logits), self.gt_classes, reduction="mean")

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }


class HoiOutputs(object):
    """
    A class that stores information about outputs of a HOI head.
    """
    def __init__(self, pred_class_logits, hopairs, pos_weights, weights=None):
        """
        Args:
            pred_class_logits (Tensor): A tensor of shape (R, K) storing the predicted
                action class logits for all R human-object pair instances.
                Each row corresponds to a human-object pair in "hopairs".
            hopairs (list[Instances]): A list of N Instances, where Instances i stores the
                proposal pairs for image i. When training, each Instances must have
                ground-truth labels stored in the field "gt_actions" and "gt_classes".
                The total number of all instances must be equal to R.
        """
        self.device = pred_class_logits.device
        self.num_preds_per_image = [len(p) for p in hopairs]
        self.pred_class_logits = pred_class_logits
        self.pos_weights = pos_weights.to(self.device)
        self.image_shapes = [x.image_size for x in hopairs]
        self.weights = weights
        
        if len(hopairs):
            if hopairs[0].has("gt_actions"):
                # The following fields should exist only when training.
                self.gt_actions = cat([x.gt_actions for x in hopairs], dim=0)
                self.gt_actions = self.gt_actions.to(self.device)
            else:
                # The following fields should be available when inference.
                self.person_boxes = [x.person_boxes for x in hopairs]
                self.object_boxes = [x.object_boxes for x in hopairs]
                self.person_box_scores = [x.person_box_scores for x in hopairs]
                self.object_box_scores = [x.object_box_scores for x in hopairs]
                self.object_box_classes = [x.object_box_classes for x in hopairs]

        self._no_instances = len(hopairs) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        gt_actions = self.gt_actions.flatten()
        num_instances = gt_actions.numel()

        pred_classes = torch.sigmoid(self.pred_class_logits).flatten()

        fg_inds = (gt_actions > 0)
        num_fg = fg_inds.nonzero().numel()
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes <= 0.5).nonzero().numel()
        num_accurate = ((pred_classes > 0.5) == gt_actions).nonzero().numel()
        fg_num_accurate = (fg_pred_classes > 0.5).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("action/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("action/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("action/false_negative", num_false_negative / num_fg)

    def binary_cross_entropy_with_logits(self):
        """
        Compute the binary cross entropy loss for action classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * F.binary_cross_entropy_with_logits(
                self.pred_class_logits,
                torch.zeros(self.pred_class_logits.size(), device=self.device),
                reduction="sum",
            )
        else:
            self._log_accuracy()
            # with torch.no_grad():
            #     tmp = F.binary_cross_entropy_with_logits(
            #         self.pred_class_logits,
            #         self.gt_actions,
            #         weight=None,
            #         reduction="none",
            #         # pos_weight=self.pos_weights
            #     )
            #     pos_loss = torch.mul(tmp, self.gt_actions)
            #     pos_loss = torch.sum(pos_loss) / torch.sum(self.gt_actions)
            #     print(pos_loss.item(), tmp.mean().item())
            return F.binary_cross_entropy_with_logits(
                self.pred_class_logits,
                self.gt_actions,
                weight=self.weights,
                reduction="mean",
                pos_weight=self.pos_weights
            )

    def losses(self):
        """
        Compute the default losses for action classification in hoi head.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_action".
        """
        return {
            "loss_action": self.binary_cross_entropy_with_logits(),
        }
    
    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K), where Ri is the number of human-object pairs
                for image i.
        """
        probs = torch.sigmoid(self.pred_class_logits)
        return probs.split(self.num_preds_per_image, dim=0)
        

    def inference(self, score_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): Only return detections with a confidence score exceeding this
                threshold.
            topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
                all detections.
        Returns:
            instances: (list[Instances]): A list of N instances, one for each image in the batch,
                that stores the topk most confidence detections.
        """
        hoi_scores = self.predict_probs()

        instances = []

        for image_id in range(len(self.image_shapes)):
            instances_per_image = interaction_inference_single_image(
                self.image_shapes[image_id],
                self.person_boxes[image_id],
                self.object_boxes[image_id],
                self.person_box_scores[image_id],
                self.object_box_scores[image_id],
                self.object_box_classes[image_id],
                hoi_scores[image_id],
                score_thresh,
                topk_per_image
            )
            instances.append(instances_per_image)

        return instances


class BoxOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """
    def __init__(self, cfg, input_shape):
        """
        Args:
            cfg
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super(BoxOutputLayers, self).__init__()
        # fmt: off
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)
        self.num_classes           = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.cls_agnostic_bbox_reg = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta        = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.test_score_thresh     = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh       = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_topk_per_image   = cfg.TEST.DETECTIONS_PER_IMAGE
        self.zero_shot_on          = cfg.ZERO_SHOT.ZERO_SHOT_ON
        # fmt: on
        
        if isinstance(input_shape, int):  # some backward compatbility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        self.cls_score = Linear(input_size, self.num_classes + 1)
        num_bbox_reg_classes = 1 if self.cls_agnostic_bbox_reg else self.num_classes
        box_dim = len(self.box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        if self.zero_shot_on:
            self._init_zero_shot(cfg)

    def _init_zero_shot(self, cfg):
        """
        Initilize module for zero-shot inference.
        Prepare the semantic embeddings for novel classes.
        Args:
            cfg: configs.
        """
        # Known classes and novel classes.
        known_classes_from_dataset = []
        if len(cfg.DATASETS.TRAIN):
            metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
            known_classes_from_dataset = metadata.get("known_classes", [])
            
        novel_classes_from_args = cfg.ZERO_SHOT.NOVEL_CLASSES
        novel_classes_from_dataset = []
        if len(cfg.DATASETS.TEST):
            metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
            novel_classes_from_dataset = metadata.get("novel_classes", [])
                
        known_classes = known_classes_from_dataset
        novel_classes = novel_classes_from_args + novel_classes_from_dataset

        self.zero_shot_predictor = ZeroShotPredictor(cfg, known_classes, novel_classes)

    def forward(self, x):
        """
        Returns:
            Tensor: Nx(K+1) scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def inference(self, predictions, proposals):
        scores, proposal_deltas = predictions
        if self.zero_shot_on:
            scores, proposal_deltas = self.zero_shot_predictor.inference(
                scores, proposal_deltas, proposals
            )
        return BoxOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).box_inference(self.test_score_thresh, self.test_nms_thresh, self.test_topk_per_image)

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
        """
        scores, proposal_deltas = predictions
        return BoxOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).losses()

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).predict_boxes_for_gt_classes()

    def predict_boxes(self, predictions, proposals):
        scores, proposal_deltas = predictions
        return FastRCNNOutputs(
            self.box2box_transform, scores, proposal_deltas, proposals, self.smooth_l1_beta
        ).predict_boxes()


class HoiOutputLayers(nn.Module):
    """
    Two linear layers for predicting action classification scores for HOI.
    """
    @configurable
    def __init__(
        self,
        input_shape,
        num_classes,
        pos_weights,
        test_score_thresh=0.0,
        test_topk_per_image=100,
        input_dup=3,
        additional_dim=0,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            num_classes (int): number of action classes
            test_score_thresh (float): threshold to filter predictions results.
            test_topk_per_image (int): number of top predictions to produce per image.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatbility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # The prediction layer for num_classes foreground classes. The input should be
        # features from person, object and union region. Thus, the input size * 3.
        self.input_dup = input_dup
        self.cls_fc1 = Linear(input_size * self.input_dup + additional_dim, input_size)
        self.cls_score = Linear(input_size, num_classes)

        for layer in [self.cls_fc1, self.cls_score]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

        self.test_score_thresh = test_score_thresh
        self.test_topk_per_image = test_topk_per_image
        self.pos_weights = pos_weights

    @classmethod
    def from_config(cls, cfg, input_shape):
        # fmt: on
        num_classes          = cfg.MODEL.ROI_HEADS.NUM_ACTIONS
        test_score_thresh    = cfg.MODEL.ROI_HEADS.HOI_SCORE_THRESH_TEST
        test_topk_per_image  = cfg.TEST.INTERACTIONS_PER_IMAGE
        action_cls_weights   = cfg.MODEL.HOI_BOX_HEAD.ACTION_CLS_WEIGHTS
        batch_size_per_image = cfg.MODEL.ROI_HEADS.HOI_BATCH_SIZE_PER_IMAGE
        ims_per_batch        = cfg.SOLVER.IMS_PER_BATCH
        # fmt: off
        if cfg.MODEL.ROI_HEADS.POSITION_FEAT == 0:
            input_dup = 3
        elif cfg.MODEL.ROI_HEADS.POSITION_FEAT == 4:
            input_dup = 6
        else:
            input_dup = 5

        additional_dim = 0
        if cfg.MODEL.ROI_HEADS.ADD_WORD_EMB == 1:
            additional_dim = 300 * 2
        if cfg.MODEL.ROI_HEADS.ADD_WORD_EMB == 2:
            additional_dim = 300 * 3
        from .roi_heads import get_convert_matrix
        verb_to_HO_matrix, obj_to_HO_matrix = get_convert_matrix(obj_class_num=81)
        verb_to_HO_matrix = torch.from_numpy(verb_to_HO_matrix)
        # Positive weights are used to balance the instances at training.
        # Get the prior distribution from the metadata.
        pos_weights = torch.full((num_classes, ), 0.)
        for dataset in cfg.DATASETS.TRAIN:
            meta = MetadataCatalog.get(dataset)
            priors = meta.get("action_priors", None)
            if priors and num_classes != 600:
                priors = torch.as_tensor(priors) * ims_per_batch * batch_size_per_image
                pos_weights_per_dataset = torch.clamp(
                    1./priors,
                    min=action_cls_weights[0],
                    max=action_cls_weights[1],
                )
                pos_weights += pos_weights_per_dataset
            elif priors and num_classes == 600:
                pos_weights = torch.from_numpy(np.sqrt(sum(inst_nums)/inst_nums))
                pass
            else:
                pos_weights += torch.full((num_classes, ), 1.)

        if len(cfg.DATASETS.TRAIN):
            pos_weights /= len(cfg.DATASETS.TRAIN)

        return {
            "input_shape": input_shape,
            "num_classes": num_classes,
            "pos_weights": pos_weights,
            "test_score_thresh": test_score_thresh,
            "test_topk_per_image": test_topk_per_image,
            "input_dup": input_dup,
            "additional_dim": additional_dim
        }

    def forward(self, u_x, p_x, o_x):
        """
        Returns:
            Tensor: NxK scores for each human-object pair
        """
        x = torch.cat([u_x, p_x, o_x], dim=-1)
        x = F.relu(self.cls_fc1(x))
        x = self.cls_score(x)
        return x

    def losses(self, pred_class_logits, hopairs, weights=None):
        """
        Args:
            pred_class_logits: return values of :meth:`forward()`.
        """
        return HoiOutputs(pred_class_logits, hopairs, self.pos_weights, weights=weights).losses()

    def inference(self, pred_class_logits, hopairs):
        return HoiOutputs(pred_class_logits, hopairs, self.pos_weights).inference(
            self.test_score_thresh, self.test_topk_per_image
        )