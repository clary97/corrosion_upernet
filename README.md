# Corrosion Segmentation — UPerNet (ConvNeXt-Tiny)

부식(corrosion) 영역을 픽셀 단위로 분할하는 시맨틱 세그멘테이션 학습 코드입니다.
백본은 **ConvNeXt-Tiny**, 디코더는 **UPerNet**(HuggingFace `openmmlab/upernet-convnext-tiny`)을 사용합니다.

원본 데이터는 4단계 부식 등급(Good / Fair / Poor / Severe)으로 라벨링되어 있으며,
이진 모드(`--num_classes 2`)에서는 Fair·Poor·Severe를 하나의 **Corrosion** 클래스로 통합합니다.

## 클래스 정의

| 모드 | 클래스 |
|------|--------|
| 4-class | `0: Good`, `1: Fair`, `2: Poor`, `3: Severe` |
| 2-class (binary) | `0: Good`, `1: Corrosion` (Fair·Poor·Severe 통합) |

## 디렉터리 구조

```
corrosion_upernet/
└── training/
    ├── main_upernet.py   # 학습 진입점 (모델 구성 + 학습 호출)
    ├── trainer.py        # 학습/검증 루프, F1·IoU 로깅, 베스트 체크포인트 저장
    ├── datahandler.py    # 데이터셋 로딩·증강, 마스크 색상→클래스 인덱스 변환
    └── evaluate.py       # 클래스별 precision/recall/F1/IoU·혼동행렬 평가
```

데이터셋은 아래 구조를 기대합니다 (마스크는 BGR 색상으로 등급 인코딩):

```
<data_dir>/
├── Train/
│   ├── Images/   # 입력 이미지
│   └── Masks/    # 라벨 마스크 (색상별 등급)
└── Test/
    ├── Images/
    └── Masks/
```

마스크 색상 → 클래스 매핑 (BGR):

| 색상 (BGR) | 클래스 |
|------------|--------|
| `(0, 0, 0)` | Good |
| `(0, 0, 128)` | Fair |
| `(0, 128, 0)` | Poor |
| `(0, 128, 128)` | Severe |

## 설치

```bash
conda create -n segformer python=3.10
conda activate segformer
pip install torch torchvision transformers albumentations opencv-python scikit-learn numpy tqdm
```

## 학습

```bash
cd training
python main_upernet.py \
    --data_dir   /path/to/unified_corrosion_plus_roboflow \
    --exp_dir    ./stored_weights/upernet_convnext_t_binary_plus \
    --num_classes 2 \
    --epochs 40 --batch_size 4 --lr 6e-5 \
    --class_weights 0.1 0.5
```

### 주요 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--data_dir` | (필수) | `Train/`, `Test/` 를 포함한 데이터 루트 |
| `--exp_dir` | (필수) | 로그·체크포인트 저장 경로 |
| `--model_name` | `openmmlab/upernet-convnext-tiny` | HuggingFace 사전학습 모델 |
| `--num_classes` | `2` | `2`(binary) 또는 `4`(등급별) |
| `--epochs` | `40` | 학습 에폭 수 |
| `--batch_size` | `4` | 배치 크기 |
| `--lr` | `6e-5` | AdamW 학습률 (weight decay 0.01) |
| `--class_weights` | `None` | cross-entropy 클래스 가중치 (개수 = `num_classes`) |

학습 중 `log.csv` 에 에폭별 Train/Test loss·F1·IoU 가 기록되며,
Test F1 이 갱신될 때마다 `weights_{epoch}.pt` 가 저장되고
학습 종료 시 최고 성능 가중치가 `weights_best.pt` 로 저장됩니다.

> **참고**: weighted F1 은 배경(Good)의 비중이 커서 높게 보일 수 있으므로,
> 실제 성능은 아래 `evaluate.py` 의 클래스별 지표로 확인하세요.

## 평가

```bash
cd training
python evaluate.py \
    --data_dir   /path/to/test_dataset \
    --weights    ./stored_weights/upernet_convnext_t_binary_plus/weights_best.pt \
    --num_classes 2 \
    --out_csv    ./stored_weights/upernet_convnext_t_binary_plus/eval.csv
```

출력:
- 클래스별 **precision / recall / F1 / IoU / FPR / support**
- macro·weighted 평균, 픽셀 정확도, mean IoU
- 혼동행렬 (rows=실제, cols=예측)
- 위 지표를 `--out_csv` 경로에 저장 (기본: 가중치 옆 `eval_per_class.csv`)

`evaluate.py` 는 `Test/` 폴더만 있으면 동작하므로 외부 평가셋에도 그대로 사용할 수 있습니다.

## 구현 메모

- 입력 정규화는 `SegformerImageProcessor(do_resize=False, do_normalize=True)` 로 ImageNet 통계를 사용합니다 (ConvNeXt 동일).
- UPerNet 출력 로짓은 입력의 1/4 해상도이므로 손실 계산 전에 `F.interpolate` 로 원본 해상도로 업샘플링합니다.
- 데이터 증강: 좌우 반전, 소폭 회전/이동/전단/스케일 (`albumentations`).
- `datahandler.py` / `trainer.py` 는 SegFormer 학습 레포의 구성을 재사용한 것으로, 이진 통합·증강·학습 루프가 동일합니다.

## 모델 비교 (벤치마크 허브)

SegFormer(B2/B3) · DeepLabV3+ · UPerNet 교차 비교 결과는
`corrosion_segformer/report_comparison.md` **9절**에 통합되어 있습니다.
(통합셋 이진 기준 요약: UPerNet이 in-domain 최고, DeepLabV3+가 새 도메인 최고, SegFormer-B3가 성능·효율 균형)

## 라이선스 / 사전학습 모델

사전학습 백본은 [openmmlab/upernet-convnext-tiny](https://huggingface.co/openmmlab/upernet-convnext-tiny) 를 사용합니다.
