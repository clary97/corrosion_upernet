"""
UperNet (ConvNeXt-tiny) 부식 세그멘테이션 학습.
datahandler/trainer는 corrosion_segformer 것을 재사용 (이진 통합·증강·학습 루프 동일).

Usage:
    python main_upernet.py \
        --data_dir   /home/ldh/minkyung/unified_corrosion_plus_roboflow \
        --exp_dir    ./stored_weights/upernet_convnext_t_binary_plus \
        --num_classes 2 \
        --epochs 40 --batch_size 4 --lr 6e-5 \
        --class_weights 0.1 0.5
"""
import argparse
import torch
from transformers import UperNetForSemanticSegmentation, SegformerImageProcessor
from datahandler import get_dataloaders
from trainer import train_model

ID2LABEL_4 = {0: 'Good', 1: 'Fair', 2: 'Poor', 3: 'Severe'}
ID2LABEL_2 = {0: 'Good', 1: 'Corrosion'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--exp_dir', required=True)
    p.add_argument('--model_name', default='openmmlab/upernet-convnext-tiny')
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--lr', type=float, default=6e-5)
    p.add_argument('--class_weights', nargs='+', type=float, default=None)
    args = p.parse_args()

    assert args.num_classes in (2, 4)
    if args.class_weights is not None and len(args.class_weights) != args.num_classes:
        raise ValueError(f'class_weights({len(args.class_weights)}) != num_classes({args.num_classes})')

    id2label = ID2LABEL_4 if args.num_classes == 4 else ID2LABEL_2
    print(f'Model  : {args.model_name}')
    print(f'Classes: {args.num_classes} -> {list(id2label.values())}')
    print(f'Data   : {args.data_dir}')

    processor = SegformerImageProcessor(do_resize=False, do_normalize=True)  # ImageNet 정규화 (ConvNeXt 동일)
    model = UperNetForSemanticSegmentation.from_pretrained(
        args.model_name, num_labels=args.num_classes, ignore_mismatched_sizes=True)

    dataloaders = get_dataloaders(args.data_dir, processor, args.batch_size,
                                  num_classes=args.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_model(model=model, dataloaders=dataloaders, optimizer=optimizer,
                bpath=args.exp_dir, class_weights=args.class_weights,
                num_epochs=args.epochs, num_classes=args.num_classes)


if __name__ == '__main__':
    main()
