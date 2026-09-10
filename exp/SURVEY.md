# 文献调研(2026-09-10,聚焦:训练自由 · 严格 1 步 · 身份保持;以及"初始噪声/结构"方向)

四组关键词 × arXiv/OpenReview/HF/GitHub。每篇:做到几步 / 要不要训练 / 对 1 步说了什么 / 对我们的含义。

## A. 单步/少步 + 身份保持

| 论文 | 步数 | 训练? | 对 1 步的说法 | 对我们的含义 |
|---|---|---|---|---|
| **OPAD** 2510.20512 | 1(SD-Turbo;Hyper-SD1.5 更差) | 是(student 与 teacher 联合训练,单卡 A40) | "现有方法在 1 步上一致失败";机制:few-step 模型用**分布匹配**训练,不保留逐轨迹的 noise→image 映射,靠轨迹训练的 adapter 不迁移。**IP-Adapter @ SDXL+TCD 1 步:DINO 0.325**;OPAD 1 步 DINO 0.637 | 我们的 1 步 DINO 0.3–0.4 与其基线一致;他们的解释是"映射粒度"而非"Q 缺结构",可作对立假设写进论文 |
| **When Few Steps Are Enough** 2605.09460 | 4–8(FLUX.schnell) | 否(冻结 InfuseNet) | 4–8 步足够,1 步未做 | 已知 |
| **FastFace** 2505.21144 | 4(SDXL 蒸馏系) | 否 | Limitations:1 步效果差,future work | 已知 |
| **SwiftPie** 2605.01510 | 1 | 是 | "first one-step subject-driven" | 已知 |
| **SpatialID(Inject Where It Matters)** 2602.13994 | 多步 | 否 | 无;mask 来自 cross-attention 响应 + 从高斯先验到 attention mask 的时间调度 | 依赖多步演化,1 步无演化;但"先高斯先验后 attention"的思路与我们 F4/S 的 mask 失败互相印证 |
| **DVI** 2512.18964 | 多步 | 否 | 无 | 语义/视觉身份解耦,评测用 |
| **Identity-Conditioned LCD** 2608.31053 | 2–3(Arc2Face → LCM) | 是(蒸馏) | 用身份嵌入替代文本条件做一致性蒸馏,2–3 步 | 即使把身份条件**训进** CM,仍需 2–3 步——旁证 1 步的困难是结构性的 |
| **PuLID** 2404.16022 | 4(SDXL-Lightning) | 是 | Lightning T2I 分支从纯噪声 4 步,ID loss 在测试条件下算;评测全部 4 步 | 9 组合里的 PuLID 一格:它为 4 步设计,1 步是它的分布外 |
| 社区:DMD2 issue #79 | 1 | — | 人脸 1 步出现"胡椒噪点",作者建议用 few-step 变体 | 与我们 1 步图的噪声残留一致;DMD2 1 步人脸本身就弱 |

## B. 初始噪声与结构(我们 G/K/O/R/T 的对照)

| 论文 | 要点 | 对我们的含义 |
|---|---|---|
| **InitNO** 2404.04650 | 逐样本优化初始噪声进入"有效区域"(≤10 步优化×≤5 次重采) | 逐样本多次前向,不满足 ≤1 NFE;但证实"有效噪声"存在且可被找到 |
| **Golden Noise / NPNet** 2411.09502(ICCV25) | 学习一个噪声→黄金噪声网络(输入:噪声 + 文本嵌入;SVD 奇异值预测 + ViT 残差),+0.4 s/+500 MB;黄金噪声是随机噪声的**小扰动**,奇异向量几乎不变。GitHub 声明权重可用于 Lightning/LCM/PCM;论文最低测到 4 步(DreamShaper-turbo),**未测 1 步**,未测身份/adapter | 学习式、外置、不过 UNet 的"结构化噪声"——正是 G 想做而做错的东西(G 破坏了先验)。可作方案 AC 测试;但它是**训练过的组件**,论文里要明确标注 |
| **NoiseAR** 2506.01337 | 自回归学习初始噪声先验(逐 patch),引入学习到的空间结构 | 同上,训练式先验 |
| **MoNO** 2607.23937 | 在 SDXL-Turbo / DMD2 **1 步**上做噪声优化恢复多样性:只在**低频分量**的固定半径球面上做黎曼更新(‖z_L‖ 固定、z_H 固定,ρ=0.15),保持先验似然;用一次单步预测做代理,K=1–4 次前向 | 直接解释 G 的失败(改了高频、破坏了范数/先验)。**可派生 0-NFE 方案 AD**:把 ε 的低频分量在球面上朝某个结构源(狗/参考的低频)旋转 |
| **Decomposable Probe** 2607.03256 | few-step 模型对 prompt/latent/score 的选择性:latent 敏感性"指纹"只在 rectified-flow 模型(SD3.5/FLUX)上持续存在,ε-预测模型(SDXL/SD1.5)没有;LCM/Flash 在 score 层有低强度尖峰;作者明确"不外推到 1 步" | ① 我们的 backbone(SDXL ε-预测)对 latent 的敏感性天然弱——与"起点结构只带来 +0.08"一致;② 换 FLUX/SD3.5 系的蒸馏模型,起点结构可能更有效——backbone 选择的依据 |

## C. 注意力侧结构注入(我们 AA 的对照)

| 论文 | 要点 | 对我们的含义 |
|---|---|---|
| **FreeControl** 2511.05219(NeurIPS25) | 训练自由结构控制:参考 latent **不加噪**,用 x̃=(1−σ)x₀(σ∈[0.25,0.5])+ 固定"关键时间步"(FLUX: 661)做**一次**前向,抽 self-attention 的 **Query** 矩阵;生成时**替换**模型的 Q(K/V 保持动态),只在**中后层**注入;+5% 成本;生成本身多步,未测 few-step | 和 AA 三处相反:注入 Q(非 K/V)、参考不加噪(非加噪)、中后层(非全层/高分辨率层)。**Q 侧注入 = H2 的直接构造**。派生方案 AB |
| MasaCtrl / Reference-only(既有) | 互注意力注入 K/V | AA 已否定其 1 步版本 |

## D. 对策略的三个结论

1. **backbone 选错了。**LCM-LoRA 是最弱的单步模型(社区共识 + Probe 论文的 ε-预测 latent 不敏感);文献里真正测 1 步的都是 **SDXL-Turbo / DMD2**(MoNO、OPAD)。9 组合矩阵应尽快扩到 Turbo/DMD2,而不是继续在 LCM-LoRA 上试方案。
2. **我们"起点结构"的发现有文献支撑但形式错了**:文献的做法是(a)保持先验的低频操作(MoNO),(b)Q 侧注入(FreeControl),(c)学习式噪声先验(NPNet)。三者都没在 1 步身份保持上试过——这是空位。
3. **"training-free 严格 1 步"的负结果有旁证**:连把身份训进 CM 的 LCD 也要 2–3 步;OPAD 明确说现有方法 1 步一致失败并给出"分布匹配丢映射"的解释。论文可以把我们的机制解释(Q 无结构)与 OPAD 的解释(映射粒度)并列为两个可区分的假设。

## E. 派生的下一批方案(按成本)

| 方案 | 来源 | 成本 | 假设(可证伪) |
|---|---|---|---|
| **AD** 低频球面旋转 | MoNO | 0 NFE | 把 ε 的低频分量(ρ=0.15)在固定范数球面上朝结构源(狗/参考 latent 低频)旋转角 θ,保持先验 → 1 步 ArcFace 升 ≥1σ 且 DINO ≤0.6 |
| **AB** Query 注入 | FreeControl | 参考一次前向(缓存)+ 1 NFE | 用 (1−σ)z_ref 在关键 t 抽 self-attn Q,生成时在中后层替换 Q → 身份 ≥ 2 步水平;控制:σ、层、t |
| **AC** NPNet 黄金噪声 | Golden Noise | 0 UNet(外置 500 MB 网络) | 学习式结构化噪声在 1 步下提升身份;需下载权重,且论文里标注为训练组件 |
| **AE** 换 backbone 重跑 O/Y | Probe、MoNO、OPAD | 与现有相同 | 在 SDXL-Turbo / DMD2 上,起点结构增益是否更大、1 步基线是否不同 |

## 来源
- https://arxiv.org/abs/2510.20512 · https://arxiv.org/abs/2605.09460 · https://arxiv.org/abs/2505.21144 · https://arxiv.org/abs/2602.13994 · https://arxiv.org/abs/2512.18964 · https://arxiv.org/abs/2608.31053 · https://arxiv.org/abs/2404.16022
- https://arxiv.org/abs/2404.04650 · https://arxiv.org/abs/2411.09502 · https://arxiv.org/abs/2506.01337 · https://arxiv.org/abs/2607.23937 · https://arxiv.org/abs/2607.03256
- https://arxiv.org/abs/2511.05219 · https://github.com/tianweiy/DMD2/issues/79 · https://github.com/xie-lab-ml/Golden-Noise-for-Diffusion-Models
