import os
import numpy as np
import h5py

# ① 파일 경로 
path = r'C:\Users\김도현\Documents\MATLAB\cap\mat_dataset\ofdm_power_dataset.mat'

# ② 출력 폴더
out_dir = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'
os.makedirs(out_dir, exist_ok=True)

# ③ .mat (v7.3, HDF5) 파일 읽기
with h5py.File(path, 'r') as f:
    # 내부 키 확인 (디버깅용)
    print("파일 내 데이터셋 목록:")
    def list_keys(g, prefix=''):
        for k, v in g.items():
            print(prefix + '/' + k, type(v))
            if isinstance(v, h5py.Group):
                list_keys(v, prefix + '/' + k)
    list_keys(f)

    # 실제 데이터 읽기
    X = np.array(f['X']).T  # MATLAB에서 (K, N)로 저장되는 경우 전치
    Y = np.array(f['Y']).T

print(f" 불러오기 완료: X{X.shape}, Y{Y.shape}")

# ④ 데이터 정제 (복소 → 전력)
if np.iscomplexobj(X):
    X = (np.abs(X) ** 2).astype(np.float32)
else:
    X = X.astype(np.float32)

Y = Y.astype(np.float32)

# ⑤ train/val/test split
N = X.shape[0]
idx = np.arange(N)
np.random.seed(42)
np.random.shuffle(idx)

train_idx = idx[:int(0.8 * N)]
val_idx   = idx[int(0.8 * N):int(0.9 * N)]
test_idx  = idx[int(0.9 * N):]

# ⑥ 분할 저장
np.save(os.path.join(out_dir, 'X_train.npy'), X[train_idx])
np.save(os.path.join(out_dir, 'Y_train.npy'), Y[train_idx])
np.save(os.path.join(out_dir, 'X_val.npy'),   X[val_idx])
np.save(os.path.join(out_dir, 'Y_val.npy'),   Y[val_idx])
np.save(os.path.join(out_dir, 'X_test.npy'),  X[test_idx])
np.save(os.path.join(out_dir, 'Y_test.npy'),  Y[test_idx])

print("저장 완료!")
print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

import h5py
path = r'C:\Users\김도현\Documents\MATLAB\cap\mat_dataset\ofdm_power_dataset.mat'

with h5py.File(path, 'r') as f:
    print("Top-level keys:", list(f.keys()))
    # 필요하면 전체 트리
    def walk(g,p=''):
        for k,v in g.items():
            print((p+'/'+k).replace('//','/'), type(v))
            if isinstance(v, h5py.Group): walk(v,p+'/'+k)
    walk(f)

import os, numpy as np, h5py

path = r'C:\Users\김도현\Documents\MATLAB\cap\mat_dataset\ofdm_power_dataset.mat'
out_dir = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'
os.makedirs(out_dir, exist_ok=True)

with h5py.File(path, 'r') as f:
    X = np.array(f['X'])   # 보통 (K, N)일 수 있음
    Y = np.array(f['Y'])

# (N,K)로 맞추기
if X.shape[0] < X.shape[1]:  # (K,N)일 때
    X = X.T
if Y.shape[0] < Y.shape[1]:
    Y = Y.T

# 복소 → 전력
if np.iscomplexobj(X):
    X = (np.abs(X)**2).astype(np.float32)
else:
    X = X.astype(np.float32)
Y = Y.astype(np.float32)

# 간단 검증
assert X.shape == Y.shape and X.ndim == 2, f"형상 확인: X{X.shape}, Y{Y.shape}"

# 스플릿
N = X.shape[0]
idx = np.arange(N); rng = np.random.default_rng(42); rng.shuffle(idx)
tr, va = int(0.8*N), int(0.9*N)
train_idx, val_idx, test_idx = idx[:tr], idx[tr:va], idx[va:]

np.save(os.path.join(out_dir,'X_train.npy'), X[train_idx])
np.save(os.path.join(out_dir,'Y_train.npy'), Y[train_idx])
np.save(os.path.join(out_dir,'X_val.npy'),   X[val_idx])
np.save(os.path.join(out_dir,'Y_val.npy'),   Y[val_idx])
np.save(os.path.join(out_dir,'X_test.npy'),  X[test_idx])
np.save(os.path.join(out_dir,'Y_test.npy'),  Y[test_idx])

print("저장 완료:", X[train_idx].shape, X[val_idx].shape, X[test_idx].shape)

import numpy as np, os
cache = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'
Xtr = np.load(os.path.join(cache,'X_train.npy'))
mu  = Xtr.mean(axis=0);  std = Xtr.std(axis=0) + 1e-6
np.save(os.path.join(cache,'mu.npy'),  mu.astype(np.float32))
np.save(os.path.join(cache,'std.npy'), std.astype(np.float32))
print("정규화 통계 저장 완료:", mu.shape)

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

# -------------------- 데이터 로드/정규화 --------------------
cache = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'
Xtr = np.load(os.path.join(cache, 'X_train.npy')); Ytr = np.load(os.path.join(cache, 'Y_train.npy'))
Xva = np.load(os.path.join(cache, 'X_val.npy'));   Yva = np.load(os.path.join(cache, 'Y_val.npy'))
mu  = np.load(os.path.join(cache, 'mu.npy'));      std = np.load(os.path.join(cache, 'std.npy'))

Xtr = (Xtr - mu) / (std + 1e-6)
Xva = (Xva - mu) / (std + 1e-6)
K = Xtr.shape[1]

class ToneDS(Dataset):
    def __init__(self, X, Y):
        self.X = X.astype(np.float32)
        self.Y = Y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i][None, :], self.Y[i]   # (1,K), (K,)

tr_loader = DataLoader(ToneDS(Xtr, Ytr), batch_size=128, shuffle=True)
va_loader = DataLoader(ToneDS(Xva, Yva), batch_size=256, shuffle=False)

# -------------------- 모델(FCN) --------------------
class FCN1D(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Conv1d(64, 1, 1)  # (B,1,K)
    def forward(self, x):
        return self.head(self.body(x)).squeeze(1)  # (B,K) logits

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FCN1D().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------------------- Focal Loss --------------------
def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = prob * targets + (1 - prob) * (1 - targets)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()

# -------------------- 학습/검증 --------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    tot = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train: opt.zero_grad()
        logits = model(x)
        loss = focal_loss(logits, y, alpha=0.75, gamma=2.0)
        if train:
            loss.backward()
            opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(loader.dataset)

for ep in range(1, 16):
    tr_loss = run_epoch(tr_loader, True)
    va_loss = run_epoch(va_loader, False)
    print(f"epoch {ep:02d} | train {tr_loss:.4f} | val {va_loss:.4f}")

# -------------------- Threshold 탐색 및 F1 --------------------
model.eval()
with torch.no_grad():
    Xb = torch.from_numpy(Xva[:, None, :]).to(device)
    p = torch.sigmoid(model(Xb)).cpu().numpy()

best_t, best_f1 = 0.5, 0.0
for t in np.linspace(0.05, 0.95, 19):
    f1 = f1_score(Yva.ravel().astype(int), (p.ravel() > t).astype(int))
    if f1 > best_f1:
        best_t, best_f1 = t, f1

print(f"Best threshold: {best_t:.2f} | Val F1 = {best_f1:.4f}")

# ==== Test 평가 + 지표 + 모델 저장 ====
import os, numpy as np, torch
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score

cache = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'
Xte = np.load(os.path.join(cache,'X_test.npy'))
Yte = np.load(os.path.join(cache,'Y_test.npy'))
mu  = np.load(os.path.join(cache,'mu.npy'))
std = np.load(os.path.join(cache,'std.npy'))

Xte = (Xte - mu) / (std + 1e-6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()

with torch.no_grad():
    Xb = torch.from_numpy(Xte[:,None,:]).to(device)
    p_test = torch.sigmoid(model(Xb)).cpu().numpy()

# 1) 검증에서 구한 최적 임계값 고정
best_t = 0.40
y_pred = (p_test.ravel() > best_t).astype(int)
y_true = Yte.ravel().astype(int)

# 2) 핵심 지표
f1  = f1_score(y_true, y_pred)
pr  = precision_score(y_true, y_pred, zero_division=0)
rc  = recall_score(y_true, y_pred, zero_division=0)
pr_auc  = average_precision_score(y_true, p_test.ravel())
try:
    roc_auc = roc_auc_score(y_true, p_test.ravel())
except ValueError:
    roc_auc = float('nan')  # 클래스 하나면 ROC-AUC 불가

print(f"[TEST] thr={best_t:.2f} | F1={f1:.4f}  P={pr:.4f}  R={rc:.4f}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

# 3) 모델 저장
os.makedirs(r'C:\Users\김도현\Documents\MATLAB\cap\results\ckpt', exist_ok=True)
torch.save(model.state_dict(), r'C:\Users\김도현\Documents\MATLAB\cap\results\ckpt\fcn_focal.pth')
print(" 모델 저장 완료: results/ckpt/fcn_focal.pth")

import os, numpy as np, torch
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score, confusion_matrix

cache = r'C:\Users\김도현\Documents\MATLAB\cap\data\cache'  # (혹은 cache_v2)
Xte = np.load(os.path.join(cache,'X_test.npy'))
Yte = np.load(os.path.join(cache,'Y_test.npy'))
mu  = np.load(os.path.join(cache,'mu.npy'))
std = np.load(os.path.join(cache,'std.npy'))

Xte = (Xte - mu) / (std + 1e-6)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()

with torch.no_grad():
    Xb = torch.from_numpy(Xte[:, None, :]).to(device)
    p_test = torch.sigmoid(model(Xb)).cpu().numpy().ravel()

thr = 0.40
y_true = Yte.ravel().astype(int)
y_pred = (p_test > thr).astype(int)

f1  = f1_score(y_true, y_pred)
pr  = precision_score(y_true, y_pred, zero_division=0)
rc  = recall_score(y_true, y_pred, zero_division=0)
pr_auc  = average_precision_score(y_true, p_test)
roc_auc = roc_auc_score(y_true, p_test)
cm = confusion_matrix(y_true, y_pred)

print(f"[TEST] thr={thr:.2f} | F1={f1:.4f}  P={pr:.4f}  R={rc:.4f}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")
print("Confusion Matrix:\n", cm)

import os, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve, roc_curve, average_precision_score,
                             roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score)

# 준비
os.makedirs(r'C:\Users\김도현\Documents\MATLAB\cap\results\figures', exist_ok=True)
figdir = r'C:\Users\김도현\Documents\MATLAB\cap\results\figures'

# 평탄화
y_true = Yte.ravel().astype(int)
p      = p_test.ravel()
best_t = 0.40  # 검증에서 찾은 값

# 1) PR 곡선 / ROC 곡선
prec, rec, _ = precision_recall_curve(y_true, p)
fpr, tpr, _  = roc_curve(y_true, p)
pr_auc  = average_precision_score(y_true, p)
roc_auc = roc_auc_score(y_true, p)

plt.figure()
plt.plot(rec, prec)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR Curve (AP={pr_auc:.3f})')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figdir, 'pr_curve.png'), dpi=150); plt.close()

plt.figure()
plt.plot(fpr, tpr)
plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC Curve (AUC={roc_auc:.3f})')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figdir, 'roc_curve.png'), dpi=150); plt.close()

# 2) Confusion Matrix (thr 고정)
y_pred = (p > best_t).astype(int)
cm = confusion_matrix(y_true, y_pred)  # [[TN,FP],[FN,TP]]
plt.figure()
plt.imshow(cm, interpolation='nearest')
plt.title(f'Confusion Matrix @thr={best_t:.2f}')
plt.colorbar(); plt.xticks([0,1], ['0','1']); plt.yticks([0,1], ['0','1'])
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha='center', va='center')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(figdir, 'confusion_matrix.png'), dpi=150); plt.close()

# 3) 임계값-성능 곡선 (F1/Precision/Recall)
ths = np.linspace(0.05, 0.95, 37)
f1s, prs, rcs = [], [], []
for t in ths:
    yp = (p > t).astype(int)
    f1s.append(f1_score(y_true, yp))
    prs.append(precision_score(y_true, yp, zero_division=0))
    rcs.append(recall_score(y_true, yp, zero_division=0))

plt.figure()
plt.plot(ths, f1s, label='F1')
plt.plot(ths, prs, label='Precision')
plt.plot(ths, rcs, label='Recall')
plt.axvline(best_t, linestyle='--')
plt.xlabel('Threshold'); plt.ylabel('Score'); plt.title('Threshold vs Metrics')
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(figdir, 'threshold_metrics.png'), dpi=150); plt.close()

# 4) 샘플 예측 vs 라벨 (임의 3개)
rng = np.random.default_rng(0)
for i, idx in enumerate(rng.choice(len(Yte), size=3, replace=False)):
    yi = Yte[idx]
    pi = p_test[idx]
    xi = Xte[idx] if 'Xte' in globals() else None

    plt.figure(figsize=(10,3))
    if xi is not None:
        plt.plot(xi, alpha=0.6, label='Power')
    plt.plot(yi * (np.max(xi) if xi is not None else 1.0), 'g--', label='Label x scale')
    plt.plot((pi > best_t) * (np.max(xi) if xi is not None else 1.0), 'r', alpha=0.7, label='Pred x scale')
    plt.title(f'Sample {idx}  |  F1 ref thr={best_t:.2f}')
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'sample_{i+1}_pred_vs_label.png'), dpi=150); plt.close()

# 5) 요약 지표 프린트
f1  = f1_score(y_true, y_pred)
pr  = precision_score(y_true, y_pred, zero_division=0)
rc  = recall_score(y_true, y_pred, zero_division=0)
print(f"[TEST] thr={best_t:.2f} | F1={f1:.4f}  P={pr:.4f}  R={rc:.4f}  PR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")
print("Saved figures to:", figdir)

