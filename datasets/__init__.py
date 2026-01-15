# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch.utils.data
from .torchvision_datasets import CocoDetection
from .coco import build as build_coco
from pycocotools.coco import COCO


def convert_to_coco_api(ds):
    """
    AOODDetection 등 커스텀 데이터셋을 COCO API 객체로 변환합니다.
    """
    coco_ds = COCO()
    # 1. Categories 설정
    # AOODDetection의 CLASS_NAMES 순서대로 ID 부여
    categories = []
    if hasattr(ds, 'CLASS_NAMES'):
        class_names = ds.CLASS_NAMES
    elif hasattr(ds, 'classes'):
        class_names = ds.classes
    else:
        # Fallback if no names found
        class_names = [str(i) for i in range(100)]  # 임시

    for i, name in enumerate(class_names):
        categories.append({'id': i, 'name': name})

    dataset = {'images': [], 'categories': categories, 'annotations': []}
    ann_id = 1

    # 2. 이미지 및 어노테이션 정보 순회
    # AOODDetection은 .imgids 리스트를 가지고 있음
    img_ids = ds.imgids if hasattr(ds, 'imgids') else list(range(len(ds)))

    for img_id in img_ids:
        # (1) 이미지 정보 가져오기
        # load_instances는 (target_xml, instances_list)를 반환한다고 가정 (aood.py 기준)
        if hasattr(ds, 'load_instances'):
            target_xml, instances = ds.load_instances(img_id)
            h = int(target_xml['annotation']['size']['height'])
            w = int(target_xml['annotation']['size']['width'])
        else:
            # Fallback: __getitem__ 사용 (비효율적일 수 있음)
            img, target = ds[img_ids.index(img_id)]
            h, w = int(target['orig_size'][0]), int(target['orig_size'][1])
            # __getitem__은 이미 transform이 적용되었을 수 있으나
            # target['orig_size']가 있다면 원본 크기 사용 가능
            # 여기서는 AOODDetection 구조상 load_instances가 있다고 봅니다.
            raise NotImplementedError("load_instances method not found in dataset")

        # 파일명 찾기
        file_name = str(img_id)
        if hasattr(ds, 'imgids2img'):
            file_name = ds.imgids2img[img_id]

        dataset['images'].append({
            'id': img_id,
            'height': h,
            'width': w,
            'file_name': file_name
        })

        # (2) 어노테이션 정보 처리
        # 평가(eval) 모드 등에 따라 클래스 필터링/마스킹 로직 적용
        # AOODDetection 내부 로직을 그대로 따라야 함
        if hasattr(ds, 'remove_unk') and ds.remove_unk:
            instances = ds.remove_novel_instances(instances)
        elif hasattr(ds, 'is_eval') and ds.is_eval:
            instances = ds.label_per_task_novel_instances_as_unk(instances)
        else:
            # 기본적으로 unlabelled training 등에서는 novel을 unk로 처리
            if hasattr(ds, 'label_all_novel_instances_as_unk'):
                instances = ds.label_all_novel_instances_as_unk(instances)

        for inst in instances:
            # bbox는 [xmin, ymin, xmax, ymax] 형식이므로 [x, y, w, h]로 변환 필요
            bbox = inst['bbox']
            bbox_coco = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]

            ann = {
                'id': ann_id,
                'image_id': img_id,
                'category_id': inst['category_id'],
                'area': inst['area'],
                'bbox': bbox_coco,
                'iscrowd': 0,
                'ignore': 0
            }
            dataset['annotations'].append(ann)
            ann_id += 1

    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

    if isinstance(dataset, CocoDetection):
        return dataset.coco

    # [수정됨] AOODDetection 등의 커스텀 데이터셋을 위한 변환 로직 추가
    return convert_to_coco_api(dataset)


def build_dataset(image_set, cfg, multi_task_eval_id=None):
    if multi_task_eval_id is None:
        if hasattr(cfg.DATASET, 'AOOD_TASK'):
            multi_task_eval_id = cfg.DATASET.AOOD_TASK
        else:
            multi_task_eval_id = 3
    if cfg.DATASET.DATASET_FILE == 'coco':
        return build_coco(image_set, cfg)
    if cfg.DATASET.DATASET_FILE == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, cfg)
    DAOD_dataset = [
        'cityscapes_to_foggycityscapes',
        'sim10k_to_cityscapes_caronly',
        'cityscapes_to_bdd_daytime',
        'pascal_to_clipart',
    ]
    if cfg.DATASET.DATASET_FILE in DAOD_dataset:
        from .DAOD import build
        # 여기서 전달된 multi_task_eval_id (예: 3)가 DAOD.py -> aood.py 로 넘어감
        return build(image_set, cfg, multi_task_eval_id=multi_task_eval_id)
    raise ValueError(f'dataset {cfg.DATASET.DATASET_FILE} not supported')