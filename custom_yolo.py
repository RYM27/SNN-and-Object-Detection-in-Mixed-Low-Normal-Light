import json
from pathlib import Path
import torch
from tqdm import tqdm
import pdb
from torchvision import transforms
from timm.models import create_model
import cv2
import numpy as np
import os
from torch.utils.data import DataLoader, dataloader, distributed
import math
from PIL import Image

from ultralytics import YOLO
from ultralytics.yolo.v8.detect import DetectionValidator, DetectionPredictor
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import attempt_load_weights, attempt_load_one_weight
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.yolo.data.dataloaders.v5loader import create_dataloader
from ultralytics.yolo.data.dataset import YOLODataset
from ultralytics.yolo.data.build import InfiniteDataLoader, seed_worker
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first
from ultralytics.yolo.data.utils import PIN_MEMORY, RANK
from ultralytics.yolo.data.augment import (Compose, Mosaic, CopyPaste, RandomPerspective, MixUp, Albumentations, RandomHSV, 
                                           RandomFlip, Format)

from VGG.model import VGG16
from EfficientFormer.models.efficientformer_v2 import efficientformerv2_l, efficientformerv2_s2

def switch_yolo_build_dataloader(cfg, batch, img_path, stride=32, rect=False, names=None, rank=-1, mode='train', switch_model='vgg16'):
    assert mode in ['train', 'val']
    shuffle = mode == 'train'
    if cfg.rect and shuffle:
        LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = SwitchYOLODataset(
            img_path=img_path,
            imgsz=cfg.imgsz,
            batch_size=batch,
            augment=mode == 'train',  # augmentation
            hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
            rect=cfg.rect or rect,  # rectangular batches
            cache=cfg.cache or None,
            single_cls=cfg.single_cls or False,
            stride=int(stride),
            pad=0.0 if mode == 'train' else 0.5,
            prefix=colorstr(f'{mode}: '),
            use_segments=cfg.task == 'segment',
            use_keypoints=cfg.task == 'keypoint',
            names=names,
            switch_model=switch_model)

    batch = min(batch, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == 'train' else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch if batch > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if cfg.image_weights or cfg.close_mosaic else InfiniteDataLoader  # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    return loader(dataset=dataset,
                  batch_size=batch,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, 'collate_fn', None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset

def v8_transforms(dataset, imgsz, hyp, switch_model):
    pre_transform = Compose([
        Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic, border=[-imgsz // 2, -imgsz // 2]),
        CopyPaste(p=hyp.copy_paste),
        RandomPerspective(
            degrees=hyp.degrees,
            translate=hyp.translate,
            scale=hyp.scale,
            shear=hyp.shear,
            perspective=hyp.perspective,
            pre_transform=SwitchYOLOLetterBox(new_shape=(imgsz, imgsz), switch_model=switch_model),
        )])
    return Compose([
        pre_transform,
        MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
        Albumentations(p=1.0),
        RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
        RandomFlip(direction='vertical', p=hyp.flipud),
        RandomFlip(direction='horizontal', p=hyp.fliplr)])  # transforms

class SwitchYOLOLetterBox:
    """Resize image and padding for detection, instance segmentation, pose"""

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32, switch_model='vgg16'):
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.switch_model = switch_model

    def __call__(self, labels=None, image=None):
        if labels is None:
            labels = {}
        img = labels.get('img') if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop('rect_shape', self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get('ratio_pad'):
            labels['ratio_pad'] = (labels['ratio_pad'], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                 value=(114, 114, 114))  # add border
        
        # switch YOLO image transform
        switch_img = labels.get('switch_img')
        switch_img = cv2.cvtColor(switch_img, cv2.COLOR_BGR2RGB)
        switch_img = Image.fromarray(switch_img)

        if self.switch_model == 'vgg16':
            resize_switch = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.switch_model == 'efficientformerv2_l' or self.switch_model == 'efficientformerv2_s2':
            resize_switch = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        switch_img = resize_switch(switch_img)

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels['img'] = img
            labels['switch_img'] = switch_img
            labels['resized_shape'] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels"""
        labels['instances'].convert_bbox(format='xyxy')
        labels['instances'].denormalize(*labels['img'].shape[:2][::-1])
        labels['instances'].scale(*ratio)
        labels['instances'].add_padding(padw, padh)
        return labels

class SwitchYOLODataset(YOLODataset):
    def __init__(self,
                 img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=None,
                 prefix='',
                 rect=False,
                 batch_size=None,
                 stride=32,
                 pad=0.0,
                 single_cls=False,
                 use_segments=False,
                 use_keypoints=False,
                 names=None,
                 switch_model='vgg16'):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.names = names
        self.switch_model = switch_model
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super(YOLODataset, self).__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls)

    def get_label_info(self, index):
        label = self.labels[index].copy()
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['switch_img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (
            label['resized_shape'][0] / label['ori_shape'][0],
            label['resized_shape'][1] / label['ori_shape'][1],
        )  # for evaluation
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                if im is None:
                    raise FileNotFoundError(f'Image Not Found {f}')
            h0, w0 = im.shape[:2]  # orig hw
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
            return im, im, (h0, w0), im.shape[:2]  # im, im, hw_original, hw_resized
        return self.ims[i], self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, im, hw_original, hw_resized

    def build_transforms(self, hyp=None):
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp, self.switch_model)
        else:
            transforms = Compose([SwitchYOLOLetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False, switch_model=self.switch_model)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == "switch_img":
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch


class SwitchYOLO(YOLO):
    @smart_inference_mode()
    def val(self, data=None, switch_model='vgg16',  **kwargs):
        """
        Validate a model on a given dataset .

        Args:
            data (str): The dataset to validate on. Accepts all formats accepted by yolo
            **kwargs : Any other args accepted by the validators. To see all args check 'configuration' section in docs
        """
        overrides = self.overrides.copy()
        overrides['rect'] = True  # rect batches as default
        overrides.update(kwargs)
        overrides['mode'] = 'val'
        args = get_cfg(cfg=DEFAULT_CFG, overrides=overrides)
        args.data = data or args.data
        if 'task' in overrides:
            self.task = args.task
        else:
            args.task = self.task
        if args.imgsz == DEFAULT_CFG.imgsz and not isinstance(self.model, (str, Path)):
            args.imgsz = self.model.args['imgsz']  # use trained imgsz unless custom value is passed
        args.imgsz = check_imgsz(args.imgsz, max_dim=1)

        validator = SwitchYOLOValidator(args=args)
        validator(model=self.model, switch_model=switch_model)
        #self.metrics = validator.metrics

        return validator.metrics
    
    @smart_inference_mode()
    def predict(self, source=None, stream=False, switch_model='vgg16', **kwargs):
        """
        Perform prediction using the YOLO model.

        Args:
            source (str | int | PIL | np.ndarray): The source of the image to make predictions on.
                          Accepts all source types accepted by the YOLO model.
            stream (bool): Whether to stream the predictions or not. Defaults to False.
            **kwargs : Additional keyword arguments passed to the predictor.
                       Check the 'configuration' section in the documentation for all available options.

        Returns:
            (List[ultralytics.yolo.engine.results.Results]): The prediction results.
        """
        if source is None:
            source = 'https://ultralytics.com/images/bus.jpg'
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        overrides = self.overrides.copy()
        overrides['conf'] = 0.25
        overrides.update(kwargs)  # prefer kwargs
        overrides['mode'] = kwargs.get('mode', 'predict')
        assert overrides['mode'] in ['track', 'predict']
        overrides['save'] = kwargs.get('save', False)  # not save files by default
        if not self.predictor:
            self.task = overrides.get('task') or self.task
            self.predictor = SwitchYOLOPredictor(overrides=overrides)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, overrides)
        return self.predictor(source=source, stream=stream, switch_model=switch_model)

class SwitchYOLOValidator(DetectionValidator):
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, switch_model='vgg16'):
        """
        Supports validation of a pre-trained model if passed or a model being trained
        if trainer is passed (trainer gets priority).
        """
        self.switch_model = switch_model

        self.speed = {'preprocess': 0.0, 'inference': 0.0, 'loss': 0.0, 'postprocess': 0.0}

        # assign normal light model as first model
        self.prev_classifier_result = 1

        self.weight_low_light = "./yolov8l-exdark/best.pt"
        self.weight_normal_light = "./yolov8l-normal-coco/best.pt"

        self.w_low_light = str(self.weight_low_light[0] if isinstance(self.weight_low_light, list) else self.weight_low_light)
        self.w_normal_light = str(self.weight_normal_light[0] if isinstance(self.weight_normal_light, list) else self.weight_normal_light)

        self.training = trainer is not None
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            model = trainer.ema.ema or trainer.model
            self.args.half = self.device.type != 'cpu'  # force FP16 val during training
            model = model.half() if self.args.half else model.float()
            self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots = trainer.epoch == trainer.epochs - 1  # always plot final epoch
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks('on_val_start')
            assert model is not None, 'Either trainer or model is needed for validation'
            self.device = select_device(self.args.device, self.args.batch)
            self.args.half &= self.device.type != 'cpu'
            model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
            self.model = model
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            else:
                self.device = model.device
                if not pt and not jit:
                    self.args.batch = 1  # export.py models default to batch-size 1
                    LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

            if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == 'classify':
                self.data = check_cls_dataset(self.args.data)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type == 'cpu':
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch, self.switch_model)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        dt = Profile(), Profile(), Profile(), Profile()
        n_batches = len(self.dataloader)
        desc = self.get_desc()
        # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
        # which may affect classification task since this arg is in yolov5/classify/val.py.
        # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
        bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val

        # switch classifier model
        # SWITCH VGG 16
        if self.switch_model == 'vgg16':
            print("\nVGG16")
            switch = VGG16()
            switch.load_state_dict(torch.load("./VGG/results/vgg16-32-95.pt"))
            #switch.load_state_dict(torch.load("./VGG/results/letterbox-vgg16-22-95.pt"))
        
        # SWITCH EFFICIENT FORMER V2 L
        elif self.switch_model == 'efficientformerv2_l':
            print("\nefficientformerv2_l")
            switch = create_model(
                "efficientformerv2_l",
                num_classes=2,
                distillation=(False),
                pretrained=True,
                fuse=True,
            )
            checkpoint = torch.load("./EfficientFormer/efficientformerv2_l_exdark_coco/best.pth", map_location='cpu')
            switch.load_state_dict(checkpoint['model'], strict=True)

        # SWITCH EFFICIENT FORMER V2 S2
        elif self.switch_model == 'efficientformerv2_s2':
            print("\nefficientformerv2_s2")
            switch = create_model(
                "efficientformerv2_s2",
                num_classes=2,
                distillation=(False),
                pretrained=True,
                fuse=True,
            )
            checkpoint = torch.load("./EfficientFormer/efficientformerv2_s2_exdark_coco/best.pth", map_location='cpu')
            switch.load_state_dict(checkpoint['model'], strict=True)
        
        switch.to(self.device)
        switch.eval()

        # counter
        all_low = 0
        all_normal = 0

        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            self.batch_i = batch_i
            # preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # inference
            with dt[1]:
                # classify normal or low light image 

                classifier_result = switch(batch['switch_img'])
                _, classifier_result = torch.max(classifier_result.data, 1)

                if classifier_result.item() == 0:
                    all_low += 1
                elif classifier_result.item() == 1:
                    all_normal += 1

                # change yolo model weights
                if classifier_result.item() != self.prev_classifier_result:
                    # low-light
                    if classifier_result.item() == 0:
                        model, ckpt = attempt_load_one_weight(self.weight_low_light if isinstance(self.weight_low_light, list) else self.w_low_light,
                                         device=self.device,
                                         inplace=True,
                                         fuse=False)
                    # normal-light    
                    elif classifier_result.item() == 1:
                        model, ckpt = attempt_load_one_weight(self.weight_normal_light if isinstance(self.weight_normal_light, list) else self.w_normal_light,
                                         device=self.device,
                                         inplace=True,
                                         fuse=False)
                    model.eval()
                    self.prev_classifier_result = classifier_result.item()

                preds = model(batch['img'])

            # loss
            with dt[2]:
                if self.training:
                    self.loss += trainer.criterion(preds, batch)[1]

            # postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks('on_val_batch_end')
        stats = self.get_stats()
        self.check_stats(stats)
        self.print_results()
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
        self.finalize_metrics()
        self.run_callbacks('on_val_end')

        print(f"ALL LOW: {all_low}")
        print(f"ALL NORMAL: {all_normal}")

        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                        tuple(self.speed.values()))
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                    LOGGER.info(f'Saving {f.name}...')
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats
        
    def finalize_metrics(self, *args, **kwargs):
        self.metrics.speed = self.speed

    def get_dataloader(self, dataset_path, batch_size, switch_model):
        # TODO: manage splits differently
        # calculate stride - check if model is initialized
        gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
        return create_dataloader(path=dataset_path,
                                 imgsz=self.args.imgsz,
                                 batch_size=batch_size,
                                 stride=gs,
                                 hyp=vars(self.args),
                                 cache=False,
                                 pad=0.5,
                                 rect=self.args.rect,
                                 workers=self.args.workers,
                                 prefix=colorstr(f'{self.args.mode}: '),
                                 shuffle=False,
                                 seed=self.args.seed)[0] if self.args.v5loader else \
            switch_yolo_build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, names=self.data['names'],
                             mode='val', switch_model=switch_model)[0]
    
    def preprocess(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['switch_img'] = batch['switch_img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)

        nb = len(batch['img'])
        self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
                   for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

        return batch

class SwitchYOLOPredictor(DetectionPredictor):
    def __call__(self, source=None, model=None, stream=False, switch_model='vgg16'):
        if stream:
            return self.stream_inference(source, model, switch_model)
        else:
            return list(self.stream_inference(source, model, switch_model))  # merge list of Result into one

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, switch_model='vgg16'):
        self.switch_model = switch_model

        if self.args.verbose:
            LOGGER.info('')

        # setup model
        if not self.model:
            self.setup_model(model)
        # setup source every time predict is called
        self.setup_source(source if source is not None else self.args.source)

        # assign normal light model as first model
        self.prev_classifier_result = 1

        self.weight_low_light = "./yolov8l-exdark/best.pt"
        self.weight_normal_light = "./yolov8l-normal-coco/best.pt"

        self.w_low_light = str(self.weight_low_light[0] if isinstance(self.weight_low_light, list) else self.weight_low_light)
        self.w_normal_light = str(self.weight_normal_light[0] if isinstance(self.weight_normal_light, list) else self.weight_normal_light)

        # check if save_dir/ label file exists
        if self.args.save or self.args.save_txt:
            (self.save_dir / 'labels' if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        # warmup model
        if not self.done_warmup:
            self.model.warmup(imgsz=(1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            self.done_warmup = True

        self.seen, self.windows, self.dt, self.batch = 0, [], (Profile(), Profile(), Profile()), None

        # switch classifier model
        # SWITCH VGG 16
        if self.switch_model == 'vgg16':
            print("\nVGG16")
            switch = VGG16()
            switch.load_state_dict(torch.load("./VGG/results/vgg16-32-95.pt"))
            #switch.load_state_dict(torch.load("./VGG/results/letterbox-vgg16-22-95.pt"))
            resize_switch = transforms.Compose([
                transforms.Resize((227,227)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        # SWITCH EFFICIENT FORMER V2 L
        elif self.switch_model == 'efficientformerv2_l':
            print("\nefficientformerv2_l")
            switch = create_model(
                "efficientformerv2_l",
                num_classes=2,
                distillation=(False),
                pretrained=True,
                fuse=True,
            )
            checkpoint = torch.load("./EfficientFormer/efficientformerv2_l_exdark_coco/best.pth", map_location='cpu')
            switch.load_state_dict(checkpoint['model'], strict=True)
            resize_switch = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # SWITCH EFFICIENT FORMER V2 S2
        elif self.switch_model == 'efficientformerv2_s2':
            print("\nefficientformerv2_s2")
            switch = create_model(
                "efficientformerv2_s2",
                num_classes=2,
                distillation=(False),
                pretrained=True,
                fuse=True,
            )
            checkpoint = torch.load("./EfficientFormer/efficientformerv2_s2_exdark_coco/best.pth", map_location='cpu')
            switch.load_state_dict(checkpoint['model'], strict=True)
            resize_switch = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        switch.to(self.device)
        switch.eval()

        self.run_callbacks('on_predict_start')

        # counter
        all_low = 0
        all_normal = 0

        for batch in self.dataset:
            self.run_callbacks('on_predict_batch_start')
            self.batch = batch
            path, im, im0s, vid_cap, s = batch
            visualize = increment_path(self.save_dir / Path(path).stem, mkdir=True) if self.args.visualize else False

            # preprocess
            with self.dt[0]:
                im = self.preprocess(im)
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # inference
            with self.dt[1]:
                # classify normal or low light image  
                switch_img = im0s[0] if isinstance(im0s, list) else im0s
                switch_img = Image.fromarray(cv2.cvtColor(switch_img, cv2.COLOR_BGR2RGB))
                switch_img = resize_switch(switch_img)
                switch_img = switch_img.unsqueeze(0)
                switch_img = switch_img.to(self.device)
                
                classifier_result = switch(switch_img)
                _, classifier_result = torch.max(classifier_result.data, 1)

                if classifier_result.item() == 0:
                    all_low += 1
                elif classifier_result.item() == 1:
                    all_normal += 1

                # change yolo model weights
                if classifier_result.item() != self.prev_classifier_result:
                    # low-light
                    if classifier_result.item() == 0:
                        self.model, ckpt = attempt_load_one_weight(self.weight_low_light if isinstance(self.weight_low_light, list) else self.w_low_light,
                                         device=self.device,
                                         inplace=True,
                                         fuse=False)
                    # normal-light    
                    elif classifier_result.item() == 1:
                        self.model, ckpt = attempt_load_one_weight(self.weight_normal_light if isinstance(self.weight_normal_light, list) else self.w_normal_light,
                                         device=self.device,
                                         inplace=True,
                                         fuse=False)
                    self.model.eval()
                    self.prev_classifier_result = classifier_result.item()

                preds = self.model(im, augment=self.args.augment, visualize=visualize)

            # postprocess
            with self.dt[2]:
                self.results = self.postprocess(preds, im, im0s)
            self.run_callbacks('on_predict_postprocess_end')

            # visualize, save, write results
            n = len(im)
            for i in range(n):
                self.results[i].speed = {
                    'preprocess': self.dt[0].dt * 1E3 / n,
                    'inference': self.dt[1].dt * 1E3 / n,
                    'postprocess': self.dt[2].dt * 1E3 / n}
                p, im0 = (path[i], im0s[i].copy()) if self.source_type.webcam or self.source_type.from_img \
                    else (path, im0s.copy())
                p = Path(p)

                if self.args.verbose or self.args.save or self.args.save_txt or self.args.show:
                    s += self.write_results(i, self.results, (p, im, im0))

                if self.args.show:
                    self.show(p)

                if self.args.save:
                    self.save_preds(vid_cap, i, str(self.save_dir / p.name))
            self.run_callbacks('on_predict_batch_end')

            yield from self.results

            # Print time (inference-only)
            if self.args.verbose:
                LOGGER.info(f'{s}{self.dt[1].dt * 1E3:.1f}ms')

        print(f"ALL LOW: {all_low}")
        print(f"ALL NORMAL: {all_normal}")

        # Release assets
        if isinstance(self.vid_writer[-1], cv2.VideoWriter):
            self.vid_writer[-1].release()  # release final video writer

        # Print results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape '
                        f'{(1, 3, *self.imgsz)}' % t)
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")

        self.run_callbacks('on_predict_end')

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.args.half else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img


def tensor2im(image_tensor):
    image_numpy = image_tensor[0].cpu().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy *= 255
    image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
    return image_numpy