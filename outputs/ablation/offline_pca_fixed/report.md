# Offline-PCA Fixed 对照实验

passed: identical image hashes, mean, optics, budget, final prior, sampler, F seeds/count/NFE and settings

图案仅由 MIRFLICKR train 的20,000张图生成；无测试GT、O场景或在线图案调整。256使用655个方向；512使用2621个方向。
中心化PCA按训练方差降序；维纳参数从与在线方法相同的三个候选中，以物理方向对齐选择；实际执行8-bit互补曝光。
PCA图案从第1张起执行，没有Walsh前缀；中心化伪逆所用训练均值不变。

## Set11_256 (n=9)

| Method | PSNR | SSIM | PInv PSNR | eta |
|---|---:|---:|---:|---:|
| Walsh Fixed | 22.9448 | 0.64752 | 21.0302 | 0.81976 |
| Offline-PCA Fixed | 23.0982 | 0.64947 | 21.5742 | 0.83718 |
| Online Adaptive | 25.6863 | 0.73338 | 25.6013 | 0.92723 |

Offline-PCA Fixed minus Walsh Fixed: +0.1534 dB; 胜/平/负=6/0/3; 图像bootstrap 95% CI=[-0.06901074946825345, 0.4140069378668214]。
Online Adaptive minus Offline-PCA Fixed: +2.5882 dB; 胜/平/负=9/0/0; 图像bootstrap 95% CI=[1.8554224798102135, 3.3480304748211407]。
Online Adaptive minus Walsh Fixed: +2.7415 dB; 胜/平/负=9/0/0; 图像bootstrap 95% CI=[1.9806409859527019, 3.5636241983645442]。

## Set11_512 (n=2)

| Method | PSNR | SSIM | PInv PSNR | eta |
|---|---:|---:|---:|---:|
| Walsh Fixed | 16.6148 | 0.27606 | 16.2338 | 0.66575 |
| Offline-PCA Fixed | 16.7474 | 0.29783 | 16.2358 | 0.66802 |
| Online Adaptive | 18.2085 | 0.42821 | 18.1794 | 0.76527 |

Offline-PCA Fixed minus Walsh Fixed: +0.1327 dB; 胜/平/负=2/0/0; 图像bootstrap 95% CI=[0.05515772560515941, 0.2101429879384824]。
Online Adaptive minus Offline-PCA Fixed: +1.4611 dB; 胜/平/负=2/0/0; 图像bootstrap 95% CI=[0.7229078116213934, 2.199244068822914]。
Online Adaptive minus Walsh Fixed: +1.5937 dB; 胜/平/负=2/0/0; 图像bootstrap 95% CI=[0.9330507995598758, 2.254401794428073]。

## Set11_all (n=11)

| Method | PSNR | SSIM | PInv PSNR | eta |
|---|---:|---:|---:|---:|
| Walsh Fixed | 21.7939 | 0.57999 | 20.1581 | 0.79176 |
| Offline-PCA Fixed | 21.9435 | 0.58554 | 20.6036 | 0.80642 |
| Online Adaptive | 24.3267 | 0.67790 | 24.2519 | 0.89778 |

Offline-PCA Fixed minus Walsh Fixed: +0.1496 dB; 胜/平/负=8/0/3; 图像bootstrap 95% CI=[-0.031881140653056694, 0.365491876146468]。
Online Adaptive minus Offline-PCA Fixed: +2.3832 dB; 胜/平/负=11/0/0; 图像bootstrap 95% CI=[1.7092552613406113, 3.0890215748630045]。
Online Adaptive minus Walsh Fixed: +2.5328 dB; 胜/平/负=11/0/0; 图像bootstrap 95% CI=[1.8475510632004466, 3.2823761756857923]。

## 结论

在线方法相对Offline-PCA平均提升2.3832 dB，11张全部胜出。离线PCA相对Walsh仅+0.1496 dB，图像bootstrap区间含零。当前结果支持：增益无法仅由训练集固定PCA基替代Walsh解释；在线测量条件下的残差方向设计提供了额外价值。不能据此声称优于所有离线设计，也不能把差值当作严格因果分解。

## 边界

Finite offline PCA baseline, not the optimum over all fixed designs.
512 training uses fixed LANCZOS upsampling of the available MIRFLICKR-256 train images.
PCA uses randomized truncated SVD; eigenpair residuals are recorded in bank metadata.
Test image bootstrap is descriptive; n=2 for the 512 subgroup is too small for generalization claims.
Online policy and static PCA differ in both global versus conditional directions and their history; not an exact causal decomposition.
Fixed-bank QR reuse changes evaluation overhead, so stored timings are not hardware acquisition-speed comparisons.

复现：见 configs/abl_offline_pca_fixed.yaml 与 configs/abl_offline_pca_fixed_512.yaml；汇总 scripts/summarize_offline_pca.py。
