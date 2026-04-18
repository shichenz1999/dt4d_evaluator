# dt4d_evaluator — 设计方案

## 目标

一个独立的 Python 包，将 `dt4d_optimizer` 输出的拟合网格序列与 DT4D HDF5 数据集中的真实序列进行比对，计算评估指标。

**输入**
- 拟合序列 — `(T, V, 3)` float32 `.npy`，`dt4d_optimizer` 的输出
- 真实序列（ground truth）— `(T, V, 3)` float32，从 `dt4d.hdf5` 读取

**输出**

每条序列的指标，按 mode 分开存为 CSV：

| 列名 | 说明 |
|---|---|
| `animal` | 动物 ID（如 `bear3EP`） |
| `seq` | rig 来源序列名（如 `bear3EP_Idle3`） |
| `target` | 目标序列名（reconstruction 时与 `seq` 相同） |
| `l1` | 所有帧的平均逐顶点 L1 距离 |
| `l2` | 所有帧的平均逐顶点 L2 距离 |
| `chamfer_l1` | 所有帧的平均 Chamfer-L1 距离 |
| `chamfer_l2` | 所有帧的平均 Chamfer-L2 距离 |

---

## 仓库结构

```
dt4d_evaluator/
├── docs/
│   ├── plan.md          ← 英文版方案
│   └── plan_ch.md       ← 本文件
├── evaluator/
│   ├── __init__.py      ← 公开 API：l1、l2、chamfer_l1、chamfer_l2
│   └── metrics.py       ← 指标实现（NumPy + PyTorch GPU）
├── run.py               ← 单条序列评估
├── batch.py             ← 批量评估，输出 CSV
├── input/               ← gitignored
└── output/              ← gitignored，CSV 结果
```

---

## 各模块职责

### `evaluator/metrics.py`

`l1`/`l2` 使用 NumPy（快，要求顶点对应）。`chamfer_l1`/`chamfer_l2` 使用 PyTorch GPU 加速，分块计算 pairwise 距离矩阵以避免显存 OOM。

| 函数 | 签名 | 说明 |
|---|---|---|
| `l1` | `(fitted, target) → (T,)` | 逐帧平均逐顶点 L1 距离；要求顶点有对应关系 |
| `l2` | `(fitted, target) → (T,)` | 逐帧平均逐顶点 L2 距离；要求顶点有对应关系 |
| `chamfer_l1` | `(fitted, target, device) → (T,)` | 逐帧双向 Chamfer，mean of sqrt(min d²) |
| `chamfer_l2` | `(fitted, target, device) → (T,)` | 逐帧双向 Chamfer，mean of min d² |

### `evaluator/__init__.py`

```python
from .metrics import l1, l2, chamfer_l1, chamfer_l2
__all__ = ['l1', 'l2', 'chamfer_l1', 'chamfer_l2']
```

### `run.py`（单条序列）

给定一条拟合序列和对应的真实序列，计算并打印指标。

```
python run.py --fitted fitted.npy --gt gt.npy
```

参数：`--fitted`、`--gt`。

### `batch.py`（批量评估）

扫描 optimizer 的 output 目录，从 HDF5 加载真实序列，计算所有拟合序列的指标，按 mode 分开写入 CSV。

```
python batch.py \
  --fitted_dir /path/to/dt4d_optimizer/output \
  --hdf5       dt4d.hdf5 \
  --output_dir output/
```

输出四个文件：
- `output/reconstruction_results.csv` — reconstruction 每条序列的详细指标
- `output/reconstruction_summary.csv` — reconstruction 指标均值
- `output/transfer_results.csv` — transfer 每条序列的详细指标
- `output/transfer_summary.csv` — transfer 指标均值

- **reconstruction**：真实序列 = rig 对应的同一序列（从 HDF5 的 `animal/seq` 读取）
- **transfer**：真实序列 = 从同目录的 `meta.json` 读取 target key，再从 HDF5 加载
- `reconstruction/` 或 `transfer/` 文件夹不存在时自动跳过

参数：`--fitted_dir`、`--hdf5`、`--output_dir`。

---

## 评估指标

### L1

逐帧平均逐顶点 L1（Manhattan）距离。要求顶点有对应关系。

```
l1[t] = mean_v( |Δx| + |Δy| + |Δz| )
```

### L2

逐帧平均逐顶点 L2（欧氏）距离。要求顶点有对应关系。

```
l2[t] = mean_v( sqrt(Δx² + Δy² + Δz²) )
```

### Chamfer-L1

双向 Chamfer 距离，最近邻距离取欧氏距离（开根号）。不要求顶点对应关系。

```
chamfer_l1[t] = ( mean_{v} min_{u} ||fitted[t,v] - target[t,u]||_2
               +  mean_{u} min_{v} ||target[t,u] - fitted[t,v]||_2 ) / 2
```

### Chamfer-L2

双向 Chamfer 距离，最近邻距离取平方欧氏距离。不要求顶点对应关系。

```
chamfer_l2[t] = ( mean_{v} min_{u} ||fitted[t,v] - target[t,u]||²
               +  mean_{u} min_{v} ||target[t,u] - fitted[t,v]||² ) / 2
```

---

## 依赖项

- `torch` — GPU 加速 Chamfer 计算
- `numpy`
- `h5py` — 从 `dt4d.hdf5` 加载真实序列
- `pandas` — 写入 CSV 结果
- `tqdm` — 进度条显示
