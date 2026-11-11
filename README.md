# PGAdaSPI

Posterior-guided adaptive pattern design for single-pixel imaging.

PGAdaSPI uses a pretrained diffusion prior to design measurements sequentially under an incoherent intensity-propagation model. Given the acquired measurements, it samples measurement-consistent posterior images, constructs new pattern directions, and realizes them as quantized DMD patterns. After acquisition, it reconstructs the image using a posterior mean conditioned on all measurements.

The released main method is `propagation-full-pseudoinverse`. It uses a full centered pseudoinverse without ridge regularization, DiffPIR for both design and reconstruction, and 8-bit complementary exposures. Test ground truth is used only for physical simulation and offline evaluation, not for selecting patterns.

## Main results

Results at a 1% effective differential measurement ratio:

| Dataset | Images | PSNR (dB) | SSIM | Result directory |
|---|---:|---:|---:|---|
| Set11 | 11 | 24.3267 | 0.6779 | `outputs/ablation/full` |
| BSD68 | 68 | 24.4808 | 0.5877 | `outputs/generalization/bsd68_bsdv2` |

Set11 uses the MIRFLICKR-trained prior. The BSD68 main result uses the BSD-domain prior; `outputs/generalization/bsd68_mirflickr` contains the cross-dataset experiment with the MIRFLICKR prior.

PSNR above uses unclamped reconstructions. JSON records also include `psnr_clamped`. SSIM is computed after clipping to [0, 1] and scaling to [0, 255], using scikit-image with `data_range=255`. Images use the native luminance-channel evaluation protocol.

## Installation

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended. Install PyTorch for your CUDA environment, then install the remaining dependencies and this package:

```bash
git clone https://github.com/BZ-SEU/PGAdaSPI.git
cd PGAdaSPI
python -m pip install numpy scipy Pillow PyYAML matplotlib scikit-image
python -m pip install --no-deps -e .
```

All commands below run from the repository root. The experiments use GPU inference; cost depends on image size and the configured posterior-chain batch sizes.

## Pretrained models and assets

Model checkpoints are distributed separately from the Git source tree. Their download location is the repository's [Releases page](https://github.com/BZ-SEU/PGAdaSPI/releases). **The initial release assets must be uploaded before these downloads are available.** Place the checkpoints at these exact paths:

| Checkpoint | Purpose |
|---|---|
| `outputs/prior_daps_v2/final_65k.pt` | MIRFLICKR prior for Set11 and cross-dataset evaluation |
| `outputs/prior_daps_bsd68_v2/final_60k.pt` | BSD-domain prior for the BSD68 main result |

Both checkpoints use the same approximately 30.3M-parameter UNet architecture. Expected SHA256 hashes are recorded in [`assets/frozen_assets.json`](assets/frozen_assets.json).

Small runtime assets, including `assets/mean.pt`, `assets/mean_512.pt`, and `assets/mean_bsd68_481x321.pt`, are included in Git. These are training-data statistics, not model weights.

The Offline-PCA comparison additionally requires `assets/offline_pca_256.pt`, `assets/offline_pca_512.pt`, or `assets/offline_pca_bsd68_321x481.pt`, depending on the experiment. These larger assets are distributed separately; they are not required by the main method. Metadata is stored alongside them as JSON. `scripts/build_offline_pca.py --help` describes how to rebuild them from training data.

## Data

The evaluation configurations expect:

```text
data/set11_proxunroll/native/<image_id>.png
data/bsd68_proxunroll/native/<image_id>.png
data/ood_mismatched_prior/...
```

Set11 consists of nine 256×256 images and two 512×512 images. BSD68 uses the native 321×481 physical canvas rather than padding to 512×512. Preserve the supplied image identities, preprocessing, dataset manifests, and hashes when comparing with the published results.

Training requires separate MIRFLICKR/BSD training data. Dataset paths in the training manifests and preparation scripts must be adapted to the local installation; test images must not be used to fit priors or tune parameters.

## Run the main method

Run one Set11 image into a new output directory:

```bash
python scripts/run.py --template SET11_256 \
  --dataset-root data/set11_proxunroll/native \
  --image-ids cameraman --device cuda:0 --output outputs/demo
```

The public experiment configurations inherit from `configs/final.yaml`. Validate a configuration before starting a full experiment:

```bash
python scripts/run.py --config configs/gen_set11_256.yaml --check-config
```

Run the two Set11 size groups and the BSD68 main experiment:

```bash
python scripts/run.py --config configs/gen_set11_256.yaml --output outputs/reproduced/set11_256
python scripts/run.py --config configs/gen_set11_512.yaml --output outputs/reproduced/set11_512
python scripts/run.py --config configs/gen_bsd68_bsdv2.yaml --output outputs/reproduced/bsd68
```

Check `run.device` in each YAML against the available GPU. These are full experiments, not quick smoke tests. Use new output directories when rerunning released results.

For the 256×256 main configuration, acquisition starts at 64 measurements and ends at 655, with batches of 32 and a final batch of 15. Design uses 32 posterior chains and up to 32 candidate directions. Final reconstruction averages 32 measurement-consistent chains. Both sampling stages use DiffPIR with 50 configured steps. The historical configuration value `final_decoder: daps_mean` names the posterior-mean decoder; the actual sampler is controlled by the sampling configuration.

## Published results and local outputs

A fresh run saves three files per image:

- **PT:** actual positive-exposure uint8 patterns and metric history, used for replay.
- **JSON:** settings, hashes, metrics, and timing.
- **PNG:** a six-panel visualization.

The public `outputs/ablation/` and `outputs/generalization/` archives include PNG/JSON results and small reports, but **exclude per-image PT files**. This reduces the experiment archive from approximately 135.37 GiB to 0.812 GiB. The result JSONs retain their original provenance, including references to omitted PT files and historical local paths.

Published PNG/JSON files support inspection and metric aggregation. They do not contain the complete measurement patterns needed for replay, redecoding, or recomputing a pseudoinverse. To obtain these, rerun the corresponding experiment. Model checkpoints are separate and remain necessary for inference.

For locally generated complete PT/JSON/PNG results:

```bash
python scripts/summarize.py outputs/demo
```

This command is read-only by default; `--write-summary` additionally saves `summarize.json`. Its completeness check requires PT files, so it does **not** summarize the PNG/JSON-only public archive. To read the released main-result metrics directly:

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

for directory in ('outputs/ablation/full', 'outputs/generalization/bsd68_bsdv2'):
    rows = []
    for path in sorted(Path(directory).glob('*/*.json')):
        if path.stem != path.parent.name:
            continue
        record = json.loads(path.read_text())
        if record.get('status') == 'complete':
            rows.append(record['final'])
    if rows:
        print(directory, 'N=', len(rows),
              'PSNR=', mean(r['psnr'] for r in rows),
              'SSIM=', mean(r['ssim'] for r in rows))
PY
```

## Experiment index

Use `scripts/run.py --config configs/<experiment>.yaml`. `abl_*.yaml` files define comparisons and ablations; `gen_*.yaml` files define dataset and sampling-ratio experiments. Configurations ending in `_512` cover the two larger Set11 images.

| Purpose | Directories under `outputs/ablation/` |
|---|---|
| Main method and fixed patterns | `full`, `fixed`, `offline_pca_fixed`, `offline_pca_fixed_bsd68` |
| AdaSense-style comparisons | `adasense_s43`, `adasense_ddrm`, `adasense_ddrm_bsd68` |
| Pattern optimization | `random`, `greedy`, `adam` |
| Objective components | `no_shrink`, `spread0p0`, `spread0p5` |
| Binary pattern realization | `dmd_1bit_alignment` |
| Alternative samplers | `sampler_daps`, `sampler_ddrm`, `sampler_ddnm` |
| Design sampling steps | `steps10`, `steps20`, `steps100`, `steps200` |
| Final decoding steps | `F_steps10`, `F_steps20`, `F_steps100`, `F_steps200` |
| Posterior sample counts | `O8`, `O16`, `O64`, `F8`, `F16`, `F64` |
| Acquisition batch size | `B16`, `B64` |
| Mismatched priors | `ood_*_test_*`, `mismatched_prior_o32_o33` |

`outputs/generalization/` contains BSD68 main/cross-dataset/fixed results and the 4%/10% Set11-256 experiments. OOD check configurations are retained for protocol validation; their historical check outputs are omitted. OOD Bias+Spread uses O32 and Spread-only uses O33, so these are not matched-sample-count comparisons.

The 1-bit result uses the `alignment` realization configuration; it should not be interpreted as a naive zero-threshold implementation.

## Training and figures

Training entry points:

```bash
python scripts/train_prior.py --help
python scripts/train_prior_bsd68_v2.py --help
```

The released MIRFLICKR checkpoint is selected at 65k steps, and the BSD-domain checkpoint at 60k steps. Preserve the training/validation splits and checkpoint-selection protocol when retraining.

`figures/` contains the plotting scripts. Rendered figures and metadata are in `outputs/paper_figures/`; large NPZ caches are omitted from Git. Plotting scripts that replay measurements require the omitted result PT files. For example, `fig4_qualitative.py` recomputes pseudoinverse reconstructions from the `fixed` and `full` patterns.

Third-party implementations under `comparable_method/` are not included. Some comparison/plotting scripts expect their results at the original local paths; these inputs must be supplied separately. The checkpoint-selection figure also references a development archive outside this repository. Full figure regeneration therefore requires additional inputs beyond the PNG/JSON-only release.

## Repository layout

```text
src/daps_spi/       Physics, pattern design, diffusion sampling, reconstruction
scripts/           Experiment, training, asset generation, and summary tools
configs/           Shared configuration and individual experiment YAMLs
assets/            Statistics, dataset splits, asset metadata, and hashes
data/              Evaluation images and manifests
figures/           Paper plotting scripts
outputs/ablation/   Published ablation PNG/JSON results
outputs/generalization/  Published generalization PNG/JSON results
outputs/paper_figures/   Rendered figures and metadata
```

---

# PGAdaSPI 中文说明

用于单像素成像的后验引导自适应图案设计。

PGAdaSPI 在非相干强度传播模型下，利用预训练扩散先验进行序贯测量设计。给定已获取的测量，它采样满足测量一致性的后验图像，构造新的图案方向，并将其实现为量化的 DMD 图案。采集结束后，利用以全部测量为条件的后验均值重建图像。

公开的主方法为 `propagation-full-pseudoinverse`。该方法使用不带 ridge 正则化的完整中心化伪逆，在图案设计和重建阶段均使用 DiffPIR，并采用 8-bit 互补曝光。测试图像的真值仅用于物理仿真和离线评价，不参与图案选择。

## 主要结果

有效差分测量采样率为 1% 时的结果：

| 数据集 | 图像数量 | PSNR (dB) | SSIM | 结果目录 |
|---|---:|---:|---:|---|
| Set11 | 11 | 24.3267 | 0.6779 | `outputs/ablation/full` |
| BSD68 | 68 | 24.4808 | 0.5877 | `outputs/generalization/bsd68_bsdv2` |

Set11 使用在 MIRFLICKR 上训练的先验。BSD68 主结果使用 BSD 同域先验；`outputs/generalization/bsd68_mirflickr` 保存使用 MIRFLICKR 先验的跨数据集实验。

上表 PSNR 使用未截断的重建值计算。JSON 记录中还包含 `psnr_clamped`。SSIM 先将重建值截断到 [0, 1] 并缩放至 [0, 255]，再使用 scikit-image 以 `data_range=255` 计算。图像遵循原生分辨率亮度通道的评测协议。

## 安装

建议使用 Python 3.10 或更新版本，以及支持 CUDA 的 PyTorch。先安装与 CUDA 环境匹配的 PyTorch，再安装其余依赖和本项目：

```bash
git clone https://github.com/BZ-SEU/PGAdaSPI.git
cd PGAdaSPI
python -m pip install numpy scipy Pillow PyYAML matplotlib scikit-image
python -m pip install --no-deps -e .
```

以下所有命令均在仓库根目录执行。实验使用 GPU 推理，计算开销取决于图像尺寸和配置中的后验链批大小。

## 预训练模型与资产

模型检查点与 Git 源码分开分发，下载位置为仓库的 [Releases 页面](https://github.com/BZ-SEU/PGAdaSPI/releases)。**首次发布的资产上传完成后，才能通过该页面下载。** 请将检查点放置到以下准确路径：

| 检查点 | 用途 |
|---|---|
| `outputs/prior_daps_v2/final_65k.pt` | 用于 Set11 和跨数据集评测的 MIRFLICKR 先验 |
| `outputs/prior_daps_bsd68_v2/final_60k.pt` | 用于 BSD68 主结果的 BSD 同域先验 |

两份检查点使用相同的 UNet 结构，参数量约为 30.3M。预期 SHA256 哈希记录在 [`assets/frozen_assets.json`](assets/frozen_assets.json) 中。

Git 仓库包含小型运行资产，包括 `assets/mean.pt`、`assets/mean_512.pt` 和 `assets/mean_bsd68_481x321.pt`。这些文件保存训练数据统计量，并非模型权重。

Offline-PCA 对照实验还需要根据具体实验使用 `assets/offline_pca_256.pt`、`assets/offline_pca_512.pt` 或 `assets/offline_pca_bsd68_321x481.pt`。这些较大的资产单独分发，运行主方法不需要它们。对应的元数据以 JSON 格式保存在同一目录。通过 `scripts/build_offline_pca.py --help` 可查看如何从训练数据重新构建这些资产。

## 数据

评测配置使用以下数据路径：

```text
data/set11_proxunroll/native/<image_id>.png
data/bsd68_proxunroll/native/<image_id>.png
data/ood_mismatched_prior/...
```

Set11 包含九张 256×256 图像和两张 512×512 图像。BSD68 使用原生 321×481 物理画布，不补零到 512×512。与公开结果比较时，请保持所提供的图像身份、预处理、数据集清单和哈希一致。

训练需要另行准备 MIRFLICKR/BSD 训练数据。训练清单和数据准备脚本中的数据集路径需要适配本地环境；不得使用测试图像拟合先验或调整参数。

## 运行主方法

对一张 Set11 图像运行主方法，并将结果保存到新目录：

```bash
python scripts/run.py --template SET11_256 \
  --dataset-root data/set11_proxunroll/native \
  --image-ids cameraman --device cuda:0 --output outputs/demo
```

公开实验配置继承自 `configs/final.yaml`。启动完整实验前，可先校验配置：

```bash
python scripts/run.py --config configs/gen_set11_256.yaml --check-config
```

运行 Set11 的两个尺寸组和 BSD68 主实验：

```bash
python scripts/run.py --config configs/gen_set11_256.yaml --output outputs/reproduced/set11_256
python scripts/run.py --config configs/gen_set11_512.yaml --output outputs/reproduced/set11_512
python scripts/run.py --config configs/gen_bsd68_bsdv2.yaml --output outputs/reproduced/bsd68
```

请根据可用 GPU 检查各 YAML 中的 `run.device`。这些命令运行完整实验，而非快速冒烟测试。重新运行已发布实验时，请使用新的输出目录。

对于 256×256 主配置，采集从 64 次测量开始，最终达到 655 次，每批 32 次，末批 15 次。图案设计使用 32 条后验链和最多 32 个候选方向。最终重建对 32 条满足测量一致性的链取均值。两个采样阶段均使用配置为 50 步的 DiffPIR。历史配置值 `final_decoder: daps_mean` 表示后验均值解码器；实际采用的采样器由采样配置控制。

## 公开结果与本地输出

一次新运行会为每张图像保存三个文件：

- **PT：** 实际正曝光 uint8 图案和指标历史，用于重放。
- **JSON：** 设置、哈希、指标和耗时。
- **PNG：** 六子图可视化。

公开的 `outputs/ablation/` 和 `outputs/generalization/` 归档包含 PNG/JSON 结果及小型报告，但**不包含逐图 PT 文件**。这样可将实验归档体积从约 135.37 GiB 减少到 0.812 GiB。结果 JSON 保留原始来源信息，包括对未发布 PT 文件和历史本地路径的引用。

公开的 PNG/JSON 文件支持结果查看和指标汇总，但不包含重放、重新解码或重新计算伪逆所需的完整测量图案。如需这些图案，请重新运行相应实验。模型检查点单独分发，推理时仍然需要。

对于本地生成的完整 PT/JSON/PNG 结果，可运行：

```bash
python scripts/summarize.py outputs/demo
```

该命令默认只读；添加 `--write-summary` 会额外保存 `summarize.json`。其完整性检查要求 PT 文件存在，因此**不能**直接汇总仅含 PNG/JSON 的公开归档。如需直接读取已发布主结果的指标，可运行：

```bash
python - <<'PY'
import json
from pathlib import Path
from statistics import mean

for directory in ('outputs/ablation/full', 'outputs/generalization/bsd68_bsdv2'):
    rows = []
    for path in sorted(Path(directory).glob('*/*.json')):
        if path.stem != path.parent.name:
            continue
        record = json.loads(path.read_text())
        if record.get('status') == 'complete':
            rows.append(record['final'])
    if rows:
        print(directory, 'N=', len(rows),
              'PSNR=', mean(r['psnr'] for r in rows),
              'SSIM=', mean(r['ssim'] for r in rows))
PY
```

## 实验索引

使用 `scripts/run.py --config configs/<experiment>.yaml` 运行实验。`abl_*.yaml` 定义对照与消融实验；`gen_*.yaml` 定义数据集和采样率实验。以 `_512` 结尾的配置对应 Set11 中两张较大的图像。

| 用途 | `outputs/ablation/` 下的目录 |
|---|---|
| 主方法与固定图案 | `full`, `fixed`, `offline_pca_fixed`, `offline_pca_fixed_bsd68` |
| AdaSense 风格对照 | `adasense_s43`, `adasense_ddrm`, `adasense_ddrm_bsd68` |
| 图案优化 | `random`, `greedy`, `adam` |
| 目标函数组成 | `no_shrink`, `spread0p0`, `spread0p5` |
| 二值图案实现 | `dmd_1bit_alignment` |
| 其他采样器 | `sampler_daps`, `sampler_ddrm`, `sampler_ddnm` |
| 设计阶段采样步数 | `steps10`, `steps20`, `steps100`, `steps200` |
| 最终解码步数 | `F_steps10`, `F_steps20`, `F_steps100`, `F_steps200` |
| 后验样本数量 | `O8`, `O16`, `O64`, `F8`, `F16`, `F64` |
| 采集批大小 | `B16`, `B64` |
| 先验失配 | `ood_*_test_*`, `mismatched_prior_o32_o33` |

`outputs/generalization/` 包含 BSD68 主结果、跨数据集结果、固定图案结果，以及 Set11-256 上 4%/10% 采样率的实验。OOD check 配置保留用于协议核验，但不提供历史 check 输出。OOD Bias+Spread 使用 O32，Spread-only 使用 O33，因此两者不是后验样本数量完全一致的比较。

1-bit 结果使用 `alignment` 实现配置，不应将其理解为朴素的零阈值实现。

## 训练与绘图

训练入口：

```bash
python scripts/train_prior.py --help
python scripts/train_prior_bsd68_v2.py --help
```

公开的 MIRFLICKR 检查点选自第 65k 步，BSD 同域检查点选自第 60k 步。重新训练时，请保持训练/验证划分和检查点选择协议一致。

`figures/` 包含绘图脚本。已生成的图和元数据位于 `outputs/paper_figures/`；较大的 NPZ 缓存不纳入 Git。需要重放测量的绘图脚本依赖未发布的结果 PT 文件。例如，`fig4_qualitative.py` 从 `fixed` 和 `full` 图案重新计算伪逆重建。

本仓库不包含 `comparable_method/` 下的第三方实现。部分比较和绘图脚本会从原本的本地路径读取第三方结果，这些输入需要另行提供。检查点选择图还引用了仓库外的开发归档。因此，要完整重新生成所有论文图，除了公开的 PNG/JSON 之外，还需要额外输入。

## 仓库结构

```text
src/daps_spi/       物理模型、图案设计、扩散采样与重建
scripts/           实验、训练、资产生成和汇总工具
configs/           公共配置与各实验 YAML
assets/            统计量、数据划分、资产元数据和哈希
data/              评测图像与清单
figures/           论文绘图脚本
outputs/ablation/   公开的消融 PNG/JSON 结果
outputs/generalization/  公开的泛化 PNG/JSON 结果
outputs/paper_figures/   已生成的图与元数据
```
