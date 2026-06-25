"""
학습된 SegFormer 체크포인트를 Test셋에서 클래스별로 평가.

학습 시 weighted F1은 Good(배경) 비중이 커서 전체 점수가 높게 보이므로,
여기서는 클래스별 F1 / IoU / precision / recall / support 와
macro·weighted 평균, 픽셀 정확도, 혼동행렬을 함께 출력한다.

Usage:
    python evaluate.py \
        --data_dir   /home/ldh/minkyung/unified_corrosion \
        --weights    ./stored_weights/segformer_b2_4class/weights_best.pt \
        --model_name nvidia/mit-b2 \
        --num_classes 4
"""
import argparse
import csv
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import UperNetForSemanticSegmentation, SegformerImageProcessor
from datahandler import CorrosionDataset

ID2LABEL_4 = {0: 'Good', 1: 'Fair', 2: 'Poor', 3: 'Severe'}
ID2LABEL_2 = {0: 'Good', 1: 'Corrosion'}


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes):
    """배치마다 (C x C) 혼동행렬(rows=true, cols=pred)을 누적."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for batch in tqdm(loader, desc='Eval'):
        pixel_values = batch['pixel_values'].to(device)
        labels       = batch['labels'].to(device)

        logits = model(pixel_values=pixel_values).logits      # (B,C,H/4,W/4)
        upsampled = F.interpolate(
            logits, size=labels.shape[-2:],
            mode='bilinear', align_corners=False)
        pred = upsampled.argmax(dim=1)                          # (B,H,W)

        t = labels.view(-1).cpu().numpy()
        p = pred.view(-1).cpu().numpy()
        # true*C + pred 를 bincount → 혼동행렬
        idx = t * num_classes + p
        cm += np.bincount(idx, minlength=num_classes**2).reshape(num_classes, num_classes)
    return cm


def metrics_from_cm(cm):
    """혼동행렬에서 클래스별 지표 계산."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)   # 실제 픽셀 수 (true)
    pred_sum = cm.sum(axis=0).astype(np.float64)  # 예측 픽셀 수
    fp = pred_sum - tp
    fn = support - tp

    total = cm.sum()
    tn = total - support - fp           # TN = 전체 - (TP+FN) - FP

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)    # = TPR
    f1        = 2 * tp / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    fpr       = fp / (fp + tn + eps)     # FP / (전체 - support)

    # 등장하지 않는 클래스(support=0)는 NaN 처리 → 평균에서 제외
    absent = support == 0
    for arr in (precision, recall, f1, iou, fpr):
        arr[absent] = np.nan

    return {
        'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
        'fpr': fpr, 'support': support, 'pred_sum': pred_sum,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',    required=True)
    parser.add_argument('--weights',     required=True, help='state_dict (.pt) 경로')
    parser.add_argument('--model_name',  default='openmmlab/upernet-convnext-tiny')
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--out_csv',     default=None,
                        help='클래스별 지표 저장 경로 (기본: weights 옆 eval_per_class.csv)')
    args = parser.parse_args()

    assert args.num_classes in (2, 4), '--num_classes 는 2 또는 4 만 지원합니다'
    id2label = ID2LABEL_4 if args.num_classes == 4 else ID2LABEL_2
    label2id = {v: k for k, v in id2label.items()}
    names = [id2label[i] for i in range(args.num_classes)]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ── 모델 로드 ────────────────────────────────────────────────────────
    model = UperNetForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True,
    )
    state = torch.load(args.weights, map_location='cpu')
    if not isinstance(state, dict) or 'decode_head.classifier.weight' not in state:
        # 혹시 모델 객체 통째로 저장된 옛 체크포인트면 state_dict 추출
        state = state.state_dict() if hasattr(state, 'state_dict') else state
    # 체크포인트 클래스 수와 인자 일치 확인
    ckpt_c = state['decode_head.classifier.weight'].shape[0]
    assert ckpt_c == args.num_classes, (
        f'체크포인트 클래스 수({ckpt_c})와 --num_classes({args.num_classes})가 다릅니다')
    model.load_state_dict(state)
    model.to(device).eval()

    print(f'Model   : {args.model_name}')
    print(f'Weights : {args.weights}')
    print(f'Classes : {args.num_classes} → {names}\n')

    # ── Test 데이터로더만 직접 구성 (Train 폴더 없어도 동작 → 외부 평가셋 지원) ──
    test_ds = CorrosionDataset(
        os.path.join(args.data_dir, 'Test'),
        SegformerImageProcessor(do_resize=False, do_normalize=True),
        augment=False, num_classes=args.num_classes)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, drop_last=False)

    cm = confusion_matrix(model, test_loader, device, args.num_classes)
    m = metrics_from_cm(cm)

    total = cm.sum()
    pixel_acc = np.diag(cm).sum() / total

    # weighted (support 가중) — NaN 클래스 제외
    w = m['support'] / m['support'].sum()
    def wavg(a):
        valid = ~np.isnan(a)
        return np.sum(a[valid] * w[valid]) / np.sum(w[valid])

    # ── 클래스별 표 출력 ─────────────────────────────────────────────────
    print('\n클래스별 성능 (Test)')
    print('-' * 82)
    print(f'{"class":<10}{"precision":>11}{"recall":>10}{"f1":>10}{"iou":>10}'
          f'{"fpr":>10}{"support(px)":>15}')
    print('-' * 82)
    for i, nm in enumerate(names):
        print(f'{nm:<10}{m["precision"][i]:>11.4f}{m["recall"][i]:>10.4f}'
              f'{m["f1"][i]:>10.4f}{m["iou"][i]:>10.4f}{m["fpr"][i]:>10.4f}'
              f'{int(m["support"][i]):>15,d}')
    print('-' * 82)
    print(f'{"macro":<10}{np.nanmean(m["precision"]):>11.4f}'
          f'{np.nanmean(m["recall"]):>10.4f}{np.nanmean(m["f1"]):>10.4f}'
          f'{np.nanmean(m["iou"]):>10.4f}{np.nanmean(m["fpr"]):>10.4f}')
    print(f'{"weighted":<10}{wavg(m["precision"]):>11.4f}{wavg(m["recall"]):>10.4f}'
          f'{wavg(m["f1"]):>10.4f}{wavg(m["iou"]):>10.4f}{wavg(m["fpr"]):>10.4f}')
    print('-' * 82)
    print(f'pixel accuracy : {pixel_acc:.4f}')
    print(f'mean IoU (macro): {np.nanmean(m["iou"]):.4f}')

    # ── 혼동행렬 출력 (rows=true, cols=pred) ────────────────────────────
    print('\n혼동행렬 (rows=실제, cols=예측, 픽셀 수)')
    header = ' ' * 10 + ''.join(f'{nm:>12}' for nm in names)
    print(header)
    for i, nm in enumerate(names):
        print(f'{nm:<10}' + ''.join(f'{int(cm[i, j]):>12,d}' for j in range(args.num_classes)))

    # ── CSV 저장 ─────────────────────────────────────────────────────────
    out_csv = args.out_csv or os.path.join(
        os.path.dirname(os.path.abspath(args.weights)), 'eval_per_class.csv')
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(['class', 'precision', 'recall', 'f1', 'iou', 'fpr', 'support_px', 'pred_px'])
        for i, nm in enumerate(names):
            wr.writerow([nm, m['precision'][i], m['recall'][i], m['f1'][i],
                         m['iou'][i], m['fpr'][i], int(m['support'][i]), int(m['pred_sum'][i])])
        wr.writerow(['macro', np.nanmean(m['precision']), np.nanmean(m['recall']),
                     np.nanmean(m['f1']), np.nanmean(m['iou']), np.nanmean(m['fpr']), '', ''])
        wr.writerow(['weighted', wavg(m['precision']), wavg(m['recall']),
                     wavg(m['f1']), wavg(m['iou']), wavg(m['fpr']), '', ''])
        wr.writerow(['pixel_accuracy', '', '', '', pixel_acc, '', int(total), ''])
    print(f'\n저장: {out_csv}')


if __name__ == '__main__':
    main()
