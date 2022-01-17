import torch
import argparse
import numpy as np

try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None

from openpifpaf.datasets import DataModule
from openpifpaf import encoder, headmeta, metric, transforms
from openpifpaf.datasets import collate_images_anns_meta, collate_images_targets_meta
from openpifpaf.plugins.coco import CocoDataset as CocoLoader

from .constants import LANE_KEYPOINTS, LANE_SKELETON, \
    LANE_SIGMAS, LANE_POSE, LANE_SCORE_WEIGHTS

#FILE UNDER CONSTRUCTION: EVALUATION DATA LOADER IS YET TO BE WRITTEN

class ArgoversePlugin(DataModule):

    #Adding placeholders for training, validation and test set

    train_annotations = 'dataset-argoverse/train_data_anns.json'
    val_annotations = 'dataset-argoverse/val_data_anns.json'
    eval_annotations = val_annotations

    train_image_dir = 'data/images/train_data/'
    val_image_dir = 'data/images/val_data/'
    eval_image_dir = val_image_dir

    orientation_invariant = 0.0
    blur = 0.0
    augmentation = False #Setting it false for now
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    b_min = 1  # 1 pixel
    square_edge = 513
    extended_scale = False
    augmentation = True

    eval_annotation_filter = False #Setting evaluation annotation to false
    eval_long_edge = 0  # set to zero to deactivate rescaling
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()

        #Define the networks for intensity and association fields
        cif = headmeta.Cif('cif', 'lane',
                           keypoints=LANE_KEYPOINTS,
                           sigmas=LANE_SIGMAS,
                           pose=LANE_POSE,
                           draw_skeleton=LANE_SKELETON,
                           score_weights=LANE_SCORE_WEIGHTS)
        caf = headmeta.Caf('caf', 'lane',
                           keypoints=LANE_KEYPOINTS,
                           sigmas=LANE_SIGMAS,
                           pose=LANE_POSE,
                           skeleton=LANE_SKELETON)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf]

    
    @classmethod
    def cli(cls, parser:argparse.ArgumentParser):
        group = parser.add_argument_group('data module Argoverse')

        group.add_argument('--argoverse-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--argoverse-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--argoverse-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--argoverse-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--argoverse-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--argoverse-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--argoverse-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--argoverse-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--argoverse-no-augmentation',
                           dest='argoverse_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--argoverse-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--argoverse-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--argoverse-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--argoverse-bmin',
                           default=cls.b_min, type=int,
                           help='b minimum in pixels')
        group.add_argument('--argoverse-apply-local-centrality-weights',
                           dest='argoverse_apply_local_centrality',
                           default=False, action='store_true',
                           help='Weigh the CIF and CAF head during training.')
        

    @classmethod
    def configure(cls, args:argparse.Namespace):
        #Assign cli information to class variables
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory
        cls.train_annotations = args.argoverse_train_annotations
        cls.val_annotations = args.argoverse_val_annotations
        cls.eval_annotations = cls.val_annotations
        cls.train_image_dir = args.argoverse_train_image_dir
        cls.val_image_dir = args.argoverse_val_image_dir
        cls.eval_image_dir = cls.val_image_dir

        cls.square_edge = args.argoverse_square_edge
        cls.extended_scale = args.argoverse_extended_scale
        cls.orientation_invariant = args.argoverse_orientation_invariant
        cls.blur = args.argoverse_blur
        cls.augmentation = args.argoverse_augmentation  # loaded by the dest name
        cls.rescale_images = args.argoverse_rescale_images
        cls.upsample_stride = args.argoverse_upsample
        cls.min_kp_anns = args.argoverse_min_kp_anns
        cls.b_min = args.argoverse_bmin

    def _preprocess(self):
        encoders = (encoder.Cif(self.head_metas[0], bmin=self.b_min),
                    encoder.Caf(self.head_metas[1], bmin=self.b_min))

        #Transforms.Compose is chaining different transformations together
        if not self.augmentation:
            return transforms.Compose([
                transforms.NormalizeAnnotations(),
                transforms.RescaleAbsolute(self.square_edge),
                transforms.CenterPad(self.square_edge),
                transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.2 * self.rescale_images,
                            2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = transforms.RescaleRelative(
                scale_range=(0.33 * self.rescale_images,
                            1.33 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            #transforms.RandomApply(transforms.HFlip(LANE_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            transforms.RandomApply(transforms.Blur(), self.blur),
            transforms.RandomChoice(
                [transforms.RotateBy90(),
                transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.2],
            ),
            transforms.Crop(self.square_edge, use_area_of_interest=True),
            transforms.CenterPad(self.square_edge),
            transforms.MinSize(min_side=32.0),
            transforms.TRAIN_TRANSFORM,
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoLoader(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1]
        )

        return torch.utils.data.Dataloader(train_data,
        batch_size=self.batch_size, shuffle=not self.debug, pin_memory=self.pin_memory,
        num_workers=self.loader_workers, drop_last=True, collate_fn=collate_images_anns_meta)
    
    def val_loader(self):
        val_data = CocoLoader(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )

        return torch.utils.data.DataLoader(val_data,
            batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate_images_targets_meta
        )
    def metrics(self):
        return [metric.Coco(COCO(self.eval_annotations), max_per_image=20,
        category_ids=[1], iou_type='keypoints', keypoint_oks_sigmas=LANE_SIGMAS)]