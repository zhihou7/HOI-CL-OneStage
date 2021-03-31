# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table

from detectron2.evaluation.evaluator import DatasetEvaluator
from lib.data.datasets import VCOCO_OBJECTS
from detectron2.utils.logger import create_small_table, setup_logger



logger = setup_logger(name=__name__)

def instances_to_list(metadata, instances, image_id):
    """
    Dump an "Instances" object to a HICO-DET matlab format that's used for evaluation.
    Format: [[hoi_id, person box, object box, score], ...]

    Args:
        metadata ()
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []
    # Meta data
    # INTERACTION_CLASSES_TO_ID_MAP = metadata.interaction_classes_to_contiguous_id
    ACTION_CLASSES_META = metadata.action_classes
    # THING_CLASSES_META = metadata.thing_classes

    # Note that HICO-DET official evaluation uses BoxMode=XYXY_ABS
    person_boxes = instances.person_boxes.tensor.numpy()
    object_boxes = instances.object_boxes.tensor.numpy()

    person_boxes = person_boxes.tolist()
    object_boxes = object_boxes.tolist()

    scores = instances.scores.tolist()
    object_classes = instances.object_classes.tolist()
    action_classes = instances.action_classes.tolist()

    results = []
    result_dict = {}
    for person_box, object_box, object_id, action_id, score in zip(
            person_boxes, object_boxes, object_classes, action_classes, scores
    ):
        # convert action result
        # action_class_name = ACTION_CLASSES_META[action_id]
        # object_class_name = THING_CLASSES_META[object_id]
        # interaction_name = action_class_name + " " + object_class_name
        # if interaction_name in INTERACTION_CLASSES_TO_ID_MAP:
        #     interaction_id = INTERACTION_CLASSES_TO_ID_MAP[interaction_name]
        # else:
        #     # invalid human-object combinations
        #     continue

        result = [
            action_id,
            person_box[0], person_box[1], person_box[2], person_box[3],
            object_box[0], object_box[1], object_box[2], object_box[3],
            score
        ]
        results.append(result)
        if tuple(person_box) in result_dict:
            result_dict[tuple(person_box)].append(result)
        else:
            result_dict[tuple(person_box)] = [result]

    roles = [['agent', 'obj'], ['agent'], ['agent', 'instr'], ['agent', 'instr'], ['agent'], ['agent', 'obj'],
             ['agent', 'instr', 'obj'], ['agent', 'obj', 'instr'], ['agent', 'instr'], ['agent', 'instr'],
             ['agent', 'instr'], ['agent', 'obj'], ['agent', 'obj'], ['agent', 'obj'], ['agent', 'instr', 'obj'],
             ['agent'], ['agent', 'instr'], ['agent', 'instr'], ['agent', 'instr'], ['agent', 'instr'], ['agent'],
             ['agent', 'instr'], ['agent', 'obj'], ['agent', 'instr'], ['agent', 'obj'], ['agent', 'instr']]

    result_vcoco = []
    for person_box in result_dict:
        result_list = result_dict[person_box]
        dic = {}
        dic['image_id'] = image_id
        dic['person_box'] = [person_box[0], person_box[1], person_box[2], person_box[3]]
        # import ipdb;ipdb.set_trace()
        aid = action_id
        for result in result_list:
            # for each obj/action
            for rname in ['agent', 'obj', 'instr']:
                if rname == 'agent':
                    # if aid == 10:
                    #  this_agent[0, 4 + aid] = det['talk_' + rid]
                    # if aid == 16:
                    #  this_agent[0, 4 + aid] = det['work_' + rid]
                    # if (aid != 10) and (aid != 16):
                    dic[ACTION_CLASSES_META[action_id] + '_' + rname] = result[9]
                else:
                    name = ACTION_CLASSES_META[action_id] + '_' + rname
                    dic[name] = np.asarray([result[5], result[6], result[7], result[8], result[9]])

        for k in ['surf_agent', 'ski_agent', 'cut_agent', 'walk_agent', 'ride_agent',
                  'talk_on_phone_agent', 'kick_agent', 'work_on_computer_agent', 'eat_agent', 'sit_agent', 'jump_agent',
                  'lay_agent', 'drink_agent', 'carry_agent', 'throw_agent', 'smile_agent', 'look_agent', 'hit_agent',
                  'snowboard_agent', 'run_agent', 'point_agent', 'read_agent', 'hold_agent', 'skateboard_agent',
                  'stand_agent', 'catch_agent', 'surf_instr', 'ski_instr', 'cut_instr', 'walk', 'cut_obj', 'ride_instr',
                  'talk_on_phone_instr', 'kick_obj', 'work_on_computer_instr', 'eat_obj', 'sit_instr', 'jump_instr',
                  'lay_instr', 'drink_instr', 'carry_obj', 'throw_obj', 'eat_instr', 'smile', 'look_obj', 'hit_instr',
                  'hit_obj', 'snowboard_instr', 'run', 'point_instr', 'read_obj', 'hold_obj', 'skateboard_instr',
                  'stand', 'catch_obj']:
            if k not in dic:
                if k.__contains__('agent'):
                    dic[k] = 0
                else:
                    dic[k] = np.asarray([np.nan, np.nan, np.nan, np.nan,  0.])
        result_vcoco.append(dic)

    return result_vcoco

class VCOCOEvaluator(DatasetEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, using COCO's metrics and APIs.
    Evaluate human-object interaction detection using 
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instance_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._tasks = self._tasks_from_config(cfg)
        self.model_name = cfg.OUTPUT_DIR.split('/')[-1]
        self.model_name = self.model_name.replace('/', '')
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(
                f"json_file was not found in MetaDataCatalog for '{dataset_name}'."
                " Trying to convert it to COCO format ..."
            )

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset


    def reset(self):
        self._predictions = []

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("bbox",)
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("keypoints",)
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if "hoi_instances" in output:
                instances = output["hoi_instances"].to(self._cpu_device)
                # import ipdb;ipdb.set_trace()
                prediction["hoi_instances"] = instances_to_list(self._metadata, instances, input["image_id"])

            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        if "hoi_instances" in predictions[0]:
            self._eval_interactions(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_interactions(self, predictions):

        logger.info("Preparing results for VCOCO format ...")
        images = [x["image_id"] for x in predictions]
        results = []
        for item in predictions:
            results.extend(item["hoi_instances"])

        DATA_DIR = '/public/data0/users/houzhi28/data0/Data/'
        from lib.evaluation.vsrl_eval import VCOCOeval
        vcocoeval = VCOCOeval(DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json',
                                   DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json',
                                   DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')
        vcocoeval._do_eval(results, ovr_thresh=0.5, model_name=self.model_name)

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "hico_interaction_results.mat")
            file_path = os.path.abspath(file_path)
            logger.info("Saving results to {}".format(file_path))

        if not self._do_evaluation:
            logger.info("Annotations are not available for evaluation.")
            return

        logger.info("Evaluating interaction using V-COCO official MATLAB code ...")

        start = 0


    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _eval_box_proposals(self, predictions):
        """
        Evaluate the box proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYXY_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                boxes.append(prediction["proposals"].proposal_boxes.tensor.numpy())
                objectness_logits.append(prediction["proposals"].interactness_logits.numpy())

            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 500]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(predictions, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
                for sub_key in ["", "_known", "_novel"]:
                    key = "R{}{}@{:d}+IoU=0.5".format(suffix, sub_key, limit)
                    res[key] = float(stats["recalls{}".format(sub_key)][0].item() * 100)
                    print(" R{}{}@{:d}+IoU@0.5 = {:.3f}".format(suffix, sub_key, limit, res[key]))

        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["box_proposals"] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


# inspired from Detectron:
# https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L255 # noqa
def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0 ** 2, 1e5 ** 2],  # all
        [0 ** 2, 32 ** 2],  # small
        [32 ** 2, 96 ** 2],  # medium
        [96 ** 2, 1e5 ** 2],  # large
        [96 ** 2, 128 ** 2],  # 96-128
        [128 ** 2, 256 ** 2],  # 128-256
        [256 ** 2, 512 ** 2],  # 256-512
        [512 ** 2, 1e5 ** 2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    isknown_obj = []
    num_pos = 0
    num_known = 0
    num_novel = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["proposals"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = predictions.interactness_logits.sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
            if obj["iscrowd"] == 0 and obj["isactive"] == 1
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_areas = torch.as_tensor(
            [obj["area"] for obj in anno if obj["iscrowd"] == 0 and obj["isactive"] == 1]
        )
        _isknown = torch.as_tensor(
            [obj["isknown"] for obj in anno if obj["iscrowd"] == 0 and obj["isactive"] == 1]
        )

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]
        _isknown = _isknown[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = pairwise_iou(predictions.proposal_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
        isknown_obj.append(_isknown)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    #gt_overlaps, sort_ids = torch.sort(gt_overlaps)

    isknown_obj = (
        torch.cat(isknown_obj, dim=0) if len(isknown_obj) else torch.zeros(0, dtype=torch.float32)
    )
    #isknown_obj = isknown_obj[sort_ids]

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()

    # compute recall for known classes
    recalls_known = torch.zeros_like(thresholds)
    known_ids = isknown_obj.nonzero().squeeze()
    for i, t in enumerate(thresholds):
        recalls_known[i] = (gt_overlaps[known_ids] >= t).float().sum() / float(len(known_ids))
    ar_known = recalls_known.mean()

    # compute recall for novel classes
    recalls_novel = torch.zeros_like(thresholds)
    novel_ids = (isknown_obj == 0).nonzero().squeeze()
    for i, t in enumerate(thresholds):
        recalls_novel[i] = (gt_overlaps[novel_ids] >= t).float().sum() / float(len(novel_ids))
    ar_novel = recalls_novel.mean()

    return {
        "ar": ar, "ar_known": ar_known, "ar_novel": ar_novel,
        "recalls": recalls, "recalls_known": recalls_known, "recalls_novel": recalls_novel,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval