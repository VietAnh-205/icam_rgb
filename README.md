# ICAM-RBF: Instance-Conditioned Adaptation Model with RBF Distance Transform

Đây là phiên bản cải tiến của [ICAM](https://github.com/CIAM-Group/ICAM) — một phương pháp học tăng cường (RL-based) để giải bài toán tối ưu hóa tổ hợp (CVRP, TSP, ATSP). Cải tiến chính: thay thế hàm khoảng cách tuyến tính bằng **RBF (Radial Basis Function)** để model phân biệt tốt hơn giữa node gần và xa trong các bản đồ có scale khác nhau.

---

## Cải tiến so với ICAM gốc

### Vấn đề của hàm gốc

Hàm thích nghi gốc phụ thuộc tuyến tính vào khoảng cách:

```
f(N, dij) = −α · log₂N · dij
```

Điều này gây ra thiếu độ ưu tiên: khoảng cách 10 đơn vị trên bản đồ nhỏ quan trọng hơn nhiều so với bản đồ lớn, nhưng hàm tuyến tính không phân biệt được.

### Giải pháp: Single RBF

```
g(dij) = exp(−γ · dij²)

f(N, dij) = −α · log₂N · exp(−γ · dij²)
```

Tính chất:
- Output luôn trong `(0, 1]` — ổn định số học hoàn toàn
- Giảm đơn điệu: node xa hơn → attention ít hơn
- `g(0) = 1`, `g(∞) → 0`
- `γ` là learnable parameter — model tự học mức độ "nhạy cảm" với khoảng cách

---

## Cấu trúc thư mục

```
ICAM/
├── ICAM_CVRP/
│   ├── CVRPModel_ICAM.py        # ← Đã sửa: thêm RBF transform
│   ├── CVRPEnv.py               # Môi trường bài toán CVRP
│   ├── CVRPTrainer.py           # ← Đã sửa: fix batch_size=0, thêm gamma log
│   ├── CVRProblemDef.py         # Sinh dữ liệu ngẫu nhiên
│   ├── CVRPTester_SetX.py       # Tester cho benchmark Set X
│   ├── CVRPTester_SetXXL.py     # Tester cho benchmark Set XXL
│   ├── cvrp_train_vst_n100_to_n500.py  # ← Đã sửa: auto resume checkpoint
│   ├── cvrp_test_main.py        # Test trên synthetic data
│   └── cvrp_test_lib.py         # Test trên CVRPLib benchmark
├── ICAM_TSP/                    # Bài toán TSP (giữ nguyên)
├── ICAM_ATSP/                   # Bài toán ATSP (giữ nguyên)
├── ICAM_CVRPTW/                 # Bài toán CVRP + Time Windows (giữ nguyên)
├── pretrained/
│   ├── icam_cvrp.pt
│   ├── icam_tsp.pt
│   └── ...
├── data/
│   ├── cvrp/
│   │   ├── vrp100_test_lkh.txt
│   │   └── vrp1000_test_lkh.txt
│   ├── CVRPLIB_X.txt
│   └── CVRPLIB_XXL.txt
└── utils/
```

---

## Yêu cầu môi trường

```
Python    >= 3.8
PyTorch   == 2.0.1
CUDA      >= 11.7 (khuyến nghị)
numpy     == 1.24.4
tqdm
pytz
```

---

## Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/VietAnh-205/ICAM-RBF.git
cd ICAM-RBF
```

### 2. Tạo môi trường ảo

```bash
conda create -n icam-rbf python=3.8
conda activate icam-rbf
```

### 3. Cài PyTorch

```bash
# CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# CPU only
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### 4. Cài dependencies còn lại

```bash
pip install numpy==1.24.4 tqdm pytz
```

### 5. Tải dữ liệu

Tải từ [Google Drive](https://drive.google.com/drive/folders/1B2qBj8rD5apvxaWuBsjOeBStOa_bMQu9) và đặt vào thư mục `data/`:

```
data/
├── cvrp/
│   ├── vrp100_test_lkh.txt     # 10,000 instances CVRP-100
│   └── vrp1000_test_lkh.txt    # 128 instances CVRP-1000
├── CVRPLIB_X.txt               # Benchmark Set X
└── CVRPLIB_XXL.txt             # Benchmark Set XXL
```

---

## Chi tiết các thay đổi trong code

### 1. `CVRPModel_ICAM.py` — Thêm RBF transform

**Thêm hàm `rbf_transform` ngay sau phần import:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def rbf_transform(dist, gamma):
    """
    Single RBF: g(dij) = exp(−γ · dij²)
    gamma: learnable parameter, dùng softplus để đảm bảo γ > 0
    Output: tensor cùng shape, giá trị trong (0, 1]
    """
    gamma_pos = F.softplus(gamma)
    return torch.exp(-gamma_pos * dist.pow(2))
```

**Thêm `self.gamma` vào `EncoderLayer.__init__`:**

```python
class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        ...
        self.alpha = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.gamma = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # THÊM
```

**Sửa `EncoderLayer.forward()` — nhận `dist, log_scale` thay vì `negative_scale_dist`:**

```python
    def forward(self, input1, dist, log_scale):   # THAY ĐỔI SIGNATURE
        q = self.Wq(input1)
        k = self.Wk(input1)
        v = self.Wv(input1)

        rbf_dist = rbf_transform(dist, self.gamma)                    # THÊM
        alpha_dist_bias_scale = self.alpha * (-log_scale) * rbf_dist  # SỬA
        AAFM_OUT = adaptation_attention_free_module(q, k, v, alpha_dist_bias_scale)
        ...
```

**Sửa `CVRP_Encoder.forward()` — truyền `dist, log_scale` riêng:**

```python
    def forward(self, depot_xy, node_xy_demand, dist, log_scale):
        ...
        # TRƯỚC: negative_scale_dist = -1 * log_scale * dist
        #        for layer in self.layers: out = layer(out, negative_scale_dist)

        # SAU:
        for layer in self.layers:
            out = layer(out, dist, log_scale)   # THAY ĐỔI
        return out
```

**Thêm `self.gamma` vào `CVRP_Decoder.__init__`:**

```python
class CVRP_Decoder(nn.Module):
    def __init__(self, **model_params):
        ...
        self.alpha1 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.Tensor([1.]), requires_grad=True)
        self.gamma  = nn.Parameter(torch.Tensor([1.]), requires_grad=True)  # THÊM
```

**Sửa `CVRP_Decoder.forward()` — dùng RBF cho `cur_dist`:**

```python
    def forward(self, encoded_last_node, load, cur_dist, log_scale, ninf_mask):
        ...
        # THÊM: tính RBF một lần, dùng ở cả 2 chỗ
        rbf_cur_dist = rbf_transform(cur_dist, self.gamma)

        # SỬA chỗ 1 — AAFM bias:
        # TRƯỚC: alpha_adaptation_bias = -1 * self.alpha1 * log_scale * cur_dist
        alpha_adaptation_bias = -1 * self.alpha1 * log_scale * rbf_cur_dist

        AAFM_OUT = adaptation_attention_free_module(q, self.k, self.v,
                                                    alpha_adaptation_bias, ninf_mask)

        score        = torch.matmul(AAFM_OUT, self.single_head_key)
        score_scaled = score / sqrt_embedding_dim

        # SỬA chỗ 2 — compatibility score:
        # TRƯỚC: score_scaled = score_scaled - self.alpha2 * log_scale * cur_dist
        score_scaled = score_scaled - self.alpha2 * log_scale * rbf_cur_dist
        ...
```

---

### 2. `CVRPTrainer.py` — Fix batch_size=0 và thêm gamma logging

**Fix batch_size có thể bằng 0 trong Stage 2:**

```python
# TRƯỚC:
true_batch_size = int(self.trainer_params['vst_base_batch_size'] * ((100 / true_problem_size) ** 2))

# SAU:
true_batch_size = max(1, int(self.trainer_params['vst_base_batch_size'] * ((100 / true_problem_size) ** 2)))
```

**Thêm log γ cuối mỗi epoch (tùy chọn, để theo dõi quá trình học):**

```python
# Thêm vào cuối _train_one_epoch(), sau dòng log score/loss
for i, layer in enumerate(self.model.encoder.layers):
    gamma_val = F.softplus(layer.gamma).item()
    self.logger.info(f'  Encoder layer {i} γ = {gamma_val:.4f}')
self.logger.info(f'  Decoder γ = {F.softplus(self.model.decoder.gamma).item():.4f}')
```

---

### 3. `cvrp_train_vst_n100_to_n500.py` — Auto resume checkpoint

**Thêm đoạn này TRƯỚC `trainer_params`:**

```python
import glob

def find_latest_checkpoint(result_dir):
    checkpoints = glob.glob(f'{result_dir}/**/checkpoint-*.pt', recursive=True)
    if not checkpoints:
        return None, None
    latest = max(checkpoints, key=os.path.getmtime)
    epoch  = int(latest.split('checkpoint-')[1].replace('.pt', ''))
    path   = os.path.dirname(latest)
    return path, epoch

RESULT_BASE_DIR = './result_cvrp_models'
latest_path, latest_epoch = find_latest_checkpoint(RESULT_BASE_DIR)

if latest_path and latest_epoch:
    print(f'>>> Resume từ checkpoint epoch {latest_epoch}: {latest_path}')
    model_load_config = {'enable': True, 'path': latest_path, 'epoch': latest_epoch}
else:
    print('>>> Không có checkpoint, train từ đầu')
    model_load_config = {'enable': False}
```

**Thay `model_load` trong `trainer_params`:**

```python
trainer_params = {
    ...
    'model_load': model_load_config,   # THAY THẾ block model_load cũ
    ...
}
```

---

## Hướng dẫn chạy

### Train từ đầu

```bash
cd ICAM_CVRP
python cvrp_train_vst_n100_to_n500.py
```

Checkpoint lưu tại `ICAM_CVRP/result_cvrp_models/{timestamp}/checkpoint-{epoch}.pt` sau mỗi epoch.

Nếu bị ngắt giữa chừng, chạy lại lệnh trên — script tự động tìm và resume từ checkpoint mới nhất.

### Test trên synthetic data

```bash
cd ICAM_CVRP
python cvrp_test_main.py
```

### Test trên CVRPLib benchmark

Sửa `model_load` trong `cvrp_test_lib.py`:

```python
tester_params = {
    ...
    'model_load': {
        'path': './result_cvrp_models/{tên-folder-checkpoint}',
        'name': 'checkpoint-1000',   # không có đuôi .pt
    },
    ...
}
```

Sau đó chạy:

```bash
python cvrp_test_lib.py
```

Kết quả lưu tại `ICAM_CVRP/result_cvrp_test/{timestamp}/run_log.txt`.

---

## Thông số training mặc định

| Tham số | Giá trị |
|---|---|
| Problem size (train) | 100 – 500 node |
| Capacity | 50 – 100 |
| Embedding dim | 128 |
| Encoder layers | 12 |
| Total epochs | 1000 |
| Stage 1 (N=100 cố định) | Epoch 1 – 100 |
| Stage 2 (N=100~500 ngẫu nhiên) | Epoch 101 – 800 |
| Stage 3 (+ TopK loss) | Epoch 801 – 1000 |
| Learning rate | 1e-4 → 1e-5 (decay ở Stage 3) |
| Batch size (Stage 1) | 32 |
| Batch size (Stage 2) | `max(1, 32 × (100/N)²)` |
| Optimizer | Adam |
| Gradient clip | max_norm = 5.0 |

---

## Tham khảo

- Paper gốc: [Instance-Conditioned Adaptation for Large-scale Generalization of Neural Routing Solver](https://arxiv.org/abs/2405.01906)
- Repo gốc: [CIAM-Group/ICAM](https://github.com/CIAM-Group/ICAM)
- Backbone: [POMO](https://github.com/yd-kwon/POMO)
