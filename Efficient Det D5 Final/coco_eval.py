# Author: Zylo117

"""
COCO-Style Evaluations

Put images here: datasets/your_project_name/val_set_name/*.jpg
Put annotations here: datasets/your_project_name/annotations/instances_{val_set_name}.json
Put weights here: /path/to/your/weights/*.pth
Change compound_coef as needed
"""

import os
import json
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string

# ===============================================
# Function definitions
# ===============================================

def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05, compound_coef=0, params=None,
                  nms_threshold=0.5, use_cuda=True, gpu=0, use_float16=False):
    """
    Run COCO-style evaluation for given images and model
    """
    results = []
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


    all_preds = []
    all_targets = []

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path_full = os.path.join(img_path, image_info['file_name'])

        #ori_imgs, framed_imgs, framed_metas = preprocess(
          #  image_path_full, max_size=512 if compound_coef is None else [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536][compound_coef],
           # mean=params['mean'], std=params['std']
        #)
        # Get preprocessing mean/std from params safely
        if isinstance(params, dict):
            mean = params.get('mean', [0.485, 0.456, 0.406])
            std = params.get('std', [0.229, 0.224, 0.225])
        else:
            mean = getattr(params, 'mean', [0.485, 0.456, 0.406])
            std = getattr(params, 'std', [0.229, 0.224, 0.225])

        ori_imgs, framed_imgs, framed_metas = preprocess(
            image_path_full,
            max_size=input_sizes[compound_coef],
            mean=mean,
            std=std,
        )


        x = torch.from_numpy(framed_imgs[0])
        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()
        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        
        #  disable gradient tracking
        with torch.no_grad():
            features, regression, classification, anchors = model(x)
        #features, regression, classification, anchors = model(x, annotations=None)


        preds = postprocess(
            x, anchors, regression, classification,
            regressBoxes, clipBoxes,
            threshold, nms_threshold
        )

        if not preds:
            all_preds.append({'scores': [], 'class_ids': [], 'rois': []})
            all_targets.append({'image_id': image_id,
                            'boxes': torch.tensor([a['bbox'] for a in coco.loadAnns(coco.getAnnIds(imgIds=image_id))], dtype=torch.float32),
                            'labels': torch.tensor([a['category_id'] for a in coco.loadAnns(coco.getAnnIds(imgIds=image_id))], dtype=torch.int64)})
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        
        # Save predictions
        all_preds.append({
            'scores': torch.as_tensor(scores).cpu(),
            'class_ids': torch.as_tensor(class_ids).cpu(),
            'rois': torch.as_tensor(rois).cpu()
        })
        
        # Save ground truths
        anns = coco.loadAnns(coco.getAnnIds(imgIds=image_id))
        boxes = torch.tensor([a['bbox'] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.int64)
        all_targets.append({'image_id': image_id, 'boxes': boxes, 'labels': labels})

        if rois.shape[0] > 0:
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            for roi_id in range(rois.shape[0]):
                score = float(scores[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,  # COCO-style
                    'score': float(score),
                    'bbox': box.tolist(),
                }
                results.append(image_result)
                
        #  free GPU tensors for this image
        del x, features, regression, classification, anchors
        torch.cuda.empty_cache()

    # Write predictions to JSON
    filepath = f'{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

    # Save preds + targets for later analysis
    torch.save({"preds": all_preds, "targets": all_targets}, f"{set_name}_eval_preds.pth")
    print(f"Saved eval preds to {set_name}_eval_preds.pth")

    #if not len(results):
    #   raise Exception('The model does not provide any valid output. Check model architecture and data input.')
    if len(results) == 0:
        print(" Warning: No valid detections this epoch.")
        return {
            "mAP5095": 0.0,   # AP@[.5:.95] (main COCO metric)
            "mAP50": 0.0,     # AP@0.5
            "mAP75": 0.0,     # AP@0.75
            "AR": 0.0         # AR@100 (all sizes)
        }

        
    # Compute COCO metrics directly here and return them
    coco_pred = coco.loadRes(filepath)
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # === COCO eval output ===
    stats = coco_eval.stats
    return {
        "mAP5095": stats[0],   # AP@[.5:.95] (main COCO metric)
        "mAP50": stats[1],     # AP@0.5
        "mAP75": stats[2],     # AP@0.75
        "AR": stats[8]         # AR@100 (all sizes)
    }



def _eval(coco_gt, image_ids, pred_json_path):
    """
    Run COCO evaluation from JSON results
    """
    coco_pred = coco_gt.loadRes(pred_json_path)
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


# ===============================================
# Script execution (only runs when file is called directly)
# ===============================================

if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold')
    ap.add_argument('--cuda', type=boolean_string, default=True)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=boolean_string, default=False)
    ap.add_argument('--override', type=boolean_string, default=True)
    args = ap.parse_args()

    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    use_cuda = args.cuda
    gpu = args.device
    use_float16 = args.float16
    override_prev_results = args.override
    project_name = args.project
    weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

    print(f'Running coco-style evaluation on project {project_name}, weights {weights_path}...')

    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params['obj_list']

    SET_NAME = params['val_set']
    VAL_GT = os.path.join("/workspace/datasets", project_name, "annotations", f"instances_{SET_NAME}.json")
    VAL_IMGS = os.path.join("/workspace/datasets", project_name, SET_NAME)

    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)

    # Fix category IDs for KITTI
    for ann in coco_gt.dataset['annotations']:
        ann['category_id'] -= 1
    for cat in coco_gt.dataset['categories']:
        cat['id'] -= 1
    coco_gt.createIndex()

    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDetBackbone(
            compound_coef=compound_coef,
            num_classes=len(obj_list),
            ratios=eval(params['anchors_ratios']),
            scales=eval(params['anchors_scales'])
        )
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)
            if use_float16:
                model.half()

        evaluate_coco(
            VAL_IMGS, SET_NAME, image_ids, coco_gt, model,
            threshold=0.05, compound_coef=compound_coef, params=params,
            nms_threshold=nms_threshold, use_cuda=use_cuda, gpu=gpu, use_float16=use_float16
        )

    _eval(coco_gt, image_ids, f'{SET_NAME}_bbox_results.json')
