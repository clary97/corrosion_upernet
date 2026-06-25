import csv
import copy
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score


def train_model(model, dataloaders, optimizer, bpath,
                class_weights=None, num_epochs=40, num_classes=4):
    since = time.time()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    else:
        weight_tensor = None

    os.makedirs(bpath, exist_ok=True)
    fieldnames = ['epoch', 'Train_loss', 'Test_loss',
                  'Train_f1', 'Train_iou', 'Test_f1', 'Test_iou']
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_f1   = 0.0
    best_wts  = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        print(f'Epoch {epoch}/{num_epochs}')
        print('-' * 10)
        summary = {}

        for phase in ['Train', 'Test']:
            model.train() if phase == 'Train' else model.eval()
            running_loss = 0.0
            num_batches  = 0
            all_true, all_pred = [], []

            for batch in tqdm(dataloaders[phase], desc=phase):
                pixel_values = batch['pixel_values'].to(device)
                labels       = batch['labels'].to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(pixel_values=pixel_values)
                    logits  = outputs.logits          # (B, C, H/4, W/4)

                    # 원래 해상도로 업샘플
                    upsampled = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode='bilinear', align_corners=False,
                    )
                    loss = F.cross_entropy(
                        upsampled, labels, weight=weight_tensor)

                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                num_batches  += 1

                pred = upsampled.argmax(dim=1).cpu().numpy().ravel()
                true = labels.cpu().numpy().ravel()
                all_pred.append(pred)
                all_true.append(true)

            y_true = np.concatenate(all_true)
            y_pred = np.concatenate(all_pred)
            epoch_loss = running_loss / num_batches
            epoch_f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            epoch_iou  = jaccard_score(y_true, y_pred, average='weighted', zero_division=0)

            summary[f'{phase}_loss'] = epoch_loss
            summary[f'{phase}_f1']   = epoch_f1
            summary[f'{phase}_iou']  = epoch_iou
            print(f'  {phase} Loss: {epoch_loss:.4f}  F1: {epoch_f1:.4f}  IoU: {epoch_iou:.4f}')

        summary['epoch'] = epoch
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(summary)

        if summary['Test_f1'] > best_f1:
            best_f1  = summary['Test_f1']
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, os.path.join(bpath, f'weights_{epoch}.pt'))
            print(f'  ★ Best model saved (Test F1: {best_f1:.4f})')

    elapsed = time.time() - since
    print(f'Training complete in {elapsed//60:.0f}m {elapsed%60:.0f}s')
    print(f'Best Test F1: {best_f1:.4f}')

    model.load_state_dict(best_wts)
    torch.save(best_wts, os.path.join(bpath, 'weights_best.pt'))
    return model
