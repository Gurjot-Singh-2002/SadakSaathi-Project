# original author: signatrix
# adapted from https://github.com/signatrix/efficientdet/blob/master/train.py
# modified by Zylo117
#import os
#print("Running train.py from:", os.path.abspath(__file__))

import argparse
import datetime
import os
import traceback
from tqdm import tqdm

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm
from pycocotools.coco import COCO
from sklearn.metrics import precision_score, recall_score, f1_score

from coco_eval import evaluate_coco, _eval  
import csv

from backbone import EfficientDetBackbone
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string

best_loss = float('inf')
best_epoch = 0

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument("--fast_debug", action="store_true", help="Run with few samples for quick error checking")

    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold for evaluation')

    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss



def compute_precision_recall(all_preds, all_targets, num_classes):
    """
    all_preds: list of dicts {'scores': tensor, 'class_ids': tensor, 'rois': tensor}
    all_targets: list of dicts {'boxes': tensor, 'labels': tensor}
    """
    # Flatten predictions and targets
    y_true = []
    y_pred = []

    for preds, targets in zip(all_preds, all_targets):
        # Assume thresholded predictions already
        pred_classes = preds['class_ids'].cpu().numpy() if len(preds['class_ids']) > 0 else []
        true_classes = targets['labels'].cpu().numpy() if len(targets['labels']) > 0 else []

        # Multi-class presence per image
        for c in range(num_classes):
            y_true.append(1 if c in true_classes else 0)
            y_pred.append(1 if c in pred_classes else 0)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1

def train(opt):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    '''
    # === FAST DEBUG MODE (limit dataset size) ===
    if getattr(opt, "fast_debug", False):
        print("⚡ Running in FAST DEBUG mode (using 20 train + 10 val samples)...")
        training_set = torch.utils.data.Subset(training_set, range(20))
        val_set = torch.utils.data.Subset(val_set, range(10))
        
    training_generator = DataLoader(training_set, **training_params)
    val_generator = DataLoader(val_set, **val_params)
    '''
    
    # === FAST DEBUG MODE (limit dataset size) ===
    if getattr(opt, "fast_debug", False):
        print("⚡ Running in FAST DEBUG mode...")

        train_limit = min(len(training_set), 20)
        val_limit = min(len(val_set), 10)

        training_set = torch.utils.data.Subset(training_set, range(train_limit))
        val_set = torch.utils.data.Subset(val_set, range(val_limit))

        print(f"👉 Using {train_limit} training samples and {val_limit} validation samples")

    # Dataloaders
    training_generator = DataLoader(
        training_set,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=(len(training_set) > opt.batch_size),  # avoid dropping everything
        collate_fn=collater,
        num_workers=opt.num_workers
    )

    val_generator = DataLoader(
        val_set,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=(len(val_set) > opt.batch_size),
        collate_fn=collater,
        num_workers=opt.num_workers
    )



    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)
                    
                    step += 1

                    #if step % opt.save_interval == 0 and step > 0:
                     #   save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                     #   print('checkpoint...')
                    if loss + opt.es_min_delta < best_loss:
                        best_loss = loss
                        best_epoch = epoch
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_best.pth')  # keep only best
                    
                    if (epoch + 1) % 20 == 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_epoch{epoch+1}.pth')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []

                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda(non_blocking=True)
                            annot = annot.cuda(non_blocking=True)

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())
                
                # free GPU memory from this batch
                del imgs, annot, cls_loss, reg_loss, loss
                torch.cuda.empty_cache()

                cls_loss = np.mean(loss_classification_ls) if loss_classification_ls else 0
                reg_loss = np.mean(loss_regression_ls) if loss_regression_ls else 0
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                # ===================== NEW METRICS BLOCK =====================


                # Paths
                VAL_GT = f"{opt.data_path}/{params.project_name}/annotations/instances_{params.val_set}.json"
                VAL_IMGS = f"{opt.data_path}/{params.project_name}/{params.val_set}"

                # Load COCO GT
                coco_gt = COCO(VAL_GT)
                coco_gt.createIndex()

                image_ids = coco_gt.getImgIds()

                # === Evaluate model ===
                results_json = f"{params.val_set}_bbox_results.json"

                # unwrap backbone (ModelWithLoss -> EfficientDetBackbone)
                backbone_model = model.module.model if isinstance(model, CustomDataParallel) else model.model
                
                # Evaluate COCO metrics
                metrics = evaluate_coco(
                    VAL_IMGS,
                    params.val_set,
                    image_ids,
                    coco_gt,
                    backbone_model,
                    threshold=0.05,
                    compound_coef=opt.compound_coef,
                    params=params,
                    nms_threshold=opt.nms_threshold,
                    use_cuda=(params.num_gpus > 0),
                    gpu=0,
                    use_float16=False
                )

                mAP5095 = metrics["mAP5095"]
                mAP50   = metrics["mAP50"]
                mAP75   = metrics["mAP75"]
                AR      = metrics["AR"]
                
                '''
                # ===================== Compute Precision / Recall / F1 =====================
                # Load preds + targets saved by evaluate_coco
                eval_data = torch.load(f"{params.val_set}_eval_preds.pth")
                all_preds = eval_data["preds"]
                all_targets = eval_data["targets"]

                # Define a helper function to compute precision, recall, F1
                def compute_metrics(preds, targets, num_classes):
                    TP, FP, FN = 0, 0, 0
                    for p, t in zip(preds, targets):
                        if p['scores'].numel() == 0:
                            FN += t['boxes'].shape[0]
                            continue
                        for cls in range(num_classes):
                            pred_cls = (p['class_ids'] == cls)
                            target_cls = (t['labels'] == cls)
                            TP += (pred_cls & target_cls).sum().item()
                            FP += (pred_cls & ~target_cls).sum().item()
                            FN += (~pred_cls & target_cls).sum().item()
                    precision = TP / (TP + FP + 1e-8)
                    recall = TP / (TP + FN + 1e-8)
                    f1 = 2 * precision * recall / (precision + recall + 1e-8)
                    return precision, recall, f1
                    '''

                #precision, recall, f1 = compute_metrics(all_preds, all_targets, len(params.obj_list))

               # Save metrics CSV
                metrics_file = os.path.join(opt.saved_path, "metrics.csv")
                if epoch == 0 and not os.path.exists(metrics_file):
                    with open(metrics_file, "w", newline="") as f:
                        writer_csv = csv.writer(f)
                        writer_csv.writerow([
                            "epoch",
                            "train/box_loss", "train/cls_loss", "train/total_loss",
                            "val/box_loss", "val/cls_loss", "val/total_loss",
                            "metrics/mAP50-95", "metrics/mAP50", "metrics/mAP75", "metrics/AR"
                        ])

                with open(metrics_file, "a", newline="") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow([
                        epoch,
                        np.mean(loss_regression_ls), np.mean(loss_classification_ls), loss,
                        reg_loss, cls_loss, loss,
                        mAP5095, mAP50, mAP75, AR
                    ])

                # =============================================================

                model.train()


                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
