# Judge Schema 修复、性能优化与恢复验证（2026-10-12）

本节为最新结果。

用户的运行已提交 Step 11，Step 12 保存了 32 条轨迹及 31 个成功评分。缺失样本 `(001002-001, 7)` 在 temperature=0 下三次返回缺少 instruction 的 JSON；原请求仅约束 JSON 对象，重复请求无法修复确定性缺字段输出。

修复：远端采用 strict JSON Schema，必需 semantic/instruction/grounding/task/reason；reason 最多 160 字符。非法响应的重试添加结构修正反馈；评分轮次只重试尚未成功的样本，有界退避，永久 HTTP/context 错误直接退出。沿用原并发请求池，DDP 主线程每秒交换评分状态，无额外 Worker 服务。双进程 Gloo 测试将 collective timeout 设为 3 秒，远端模拟等待 4 秒并自动重试一次，总等待超过 8 秒，通信仍正常完成。

实际缺失样本的 Schema 请求成功：四维分数 4/3/4/4，57 个输出 Token，约 1.49 秒。之前 Run 的评分输出中位数 104 Token、均值 108.45，理由中位数 321 字符；恢复后新批次输出均值约 60 Token。Schema 不改变原四维评分含义和 Reward 计算，已经成功的旧评分不重做。

## 同批 7.2B 前后向性能对照

仅作前后向诊断，没有调用 Optimizer 或修改生产 checkpoint；同一份 Step 11 State、相同 Step 12 数据/Reward/Advantage、同一个 Backbone。

| 执行路径 | 前后向秒 | Teacher Forcing 位置 | 峰值 allocated bytes | Loss | Grad Norm |
|---|---:|---:|---:|---:|---:|
| 原分批、全位置 LogSoftmax | 18.776 | 64816 | 42411128320 | 0.358528 | 0.071274 |
| 长度排序、Completion LogSoftmax | 11.100 | 52240 | 42083972608 | 0.358512 | 0.071087 |

这是一轮诊断对照，不是多次重复基准；耗时约下降 41%，处理位置减少约 19.4%。排序改变 BF16 批形状/累加次序，两个实际梯度的相对 L2 差为 0.048987，不宣称组合优化逐位等价。独立 CUDA 测试确认在相同分批下，Completion-only LogSoftmax 的有效 LogProb 完全一致、完整 Initial State 梯度与原路径在 rtol=1e-5/atol=1e-6 内一致；Prompt-only chunk 的梯度链不断开，冻结 Backbone 无梯度。长度排序仍共用原全局有效 Token 分母，可用 length_bucketing=false 保留原分批顺序。

## 实际恢复至 Step 14

```bash
uv run python -m grpo.train --config configs/online.full.json --resume --stop-after-steps 14
```

| Step | Judge 秒 | Training 秒 | 总 Step 秒 | Requested / Cached Scores |
|---|---:|---:|---:|---|
| 12 | 0.955 | 12.943 | 14.033 | 1 / 31 |
| 13 | 29.209 | 10.954 | 44.845 | 32 / 0 |
| 14 | 34.924 | 17.152 | 67.736 | 32 / 0 |

不同 Step 的 Prompt/Completion 长度不同，这个表不能作为相同负载的整体加速比。Judge 在真实新批次仍是主要耗时，需要远端推理服务改善 Prefill/并发性能；本次没有改动远端配置。

逐条核对 Step 12：原 Completion 和 Old LogProb 全部保留，31 个既有 Judge 结果完全相同，仅补缺失评分。该批次使用原 Step 11 State 指纹。最终 Global Step=Policy Version=14、cursor=56、AdamW step=14，pending 已清理，未重复更新任何已提交 Step。备份位于 latest.before-judge-fix.pt / pending.before-judge-fix.json。

同一 SwanLab Run `948f384a5ded46bfa209db78f41ec4ad` 在线上传完成，本次 236 files / 1.8 MB。最终 `uv run pytest -q`：**54 passed，1 skipped**；跳过项为 NCCL 双 GPU 实机验证。新增覆盖 Schema 缺字段修复、仅重试失败项、永久错误保留同批成功项、历史失败次数、DDP 长请求心跳、Padding 下降和 Completion-only LogSoftmax 完整 BPTT。

---

# 7.2B 对齐拦截修复与真实恢复验证（2026-10-12）

本节为最新结果；后面的首次 Online 验证及历史离线记录保留当时状态。

用户的 `runs/online_7b` 在 Step 2 因最大 LogProb 误差 0.578527 中断。检查确认 checkpoint 已提交 Step=1/Policy Version=1/cursor=4，待处理批次 State 指纹完全匹配。

对 6973 个有效 Token 复算：平均 Ratio=1.000089，Clip Fraction=0.1578%，平均绝对误差=0.017777，RMS=0.030866；绝对误差 99% 分位数为 0.107010。使用原 State、所有原 seed 和现有 batched Rollout 重放，32 条 Completion 的 Token 序列完全相同，全部 6973 个采样 LogProb 误差为 0.0。原始 Old LogProb 正确；故障由 BF16 单 Token Decode 与分块 Teacher Forcing 的个别数值尾部误差触发。

修复：最大误差保留为诊断/告警指标，不再单独终止更新。平均 Ratio 偏差 0.02、Clip Fraction 1% 的整体门限保持不变，非有限数值和 State/Step/Policy Version 错配继续拒绝。没有替换 Old LogProb，没有修改 CUDA kernel、BF16 Forward/Backward 或 PPO Loss。

已备份 `latest.before-alignment-fix.pt` 和 `pending.before-alignment-fix.json`，实际执行：

```bash
uv run python -m grpo.train --config configs/online.full.json --resume --stop-after-steps 3
```

| Step | State 更新 Norm | Ratio Mean | Clip Fraction | 最大 LogProb 误差 |
|---|---:|---:|---:|---:|
| 2 | 0.191255 | 1.000089 | 0.001578 | 0.578527 |
| 3 | 0.156069 | 0.999684 | 0.001594 | 0.489635 |

恢复后的 Step 2 逐条核对原待处理数据：Completion、Old LogProb、Judge 结果均保持不变，使用原已提交 Step 1 State；Judge requested_scores=0、cached_scores=32。Step 3 完成新一轮采样/评分/更新。最终 checkpoint Global Step=3、Policy Version=3、cursor=12、AdamW 内部 step=3；三个 Step 共 96 条唯一轨迹，版本分别为 0/1/2，pending 已清理。原 Step 1 未重复更新。

同一 SwanLab Run `948f384a5ded46bfa209db78f41ec4ad` 已确认在线上传完成：本次 163 files / 1.3 MB，见 [在线训练 Run](https://swanlab.cn/@Alic/rwkv-state-grpo/runs/948f384a5ded46bfa209db78f41ec4ad)。此前未验收的云端上传与 7.2B 在线更新，现已完成三步范围的实际验证；NCCL 双 GPU 仍未验证。

最终 `uv run pytest -q`：**47 passed，1 skipped**。新增测试覆盖孤立误差允许更新并保留告警、整体 Ratio 漂移/Clip 比例超限仍拒绝更新，以及无需修改 JSON 的恢复 CLI。

继续原训练：

```bash
uv run python -m grpo.train --config configs/online.full.json --resume
```

---

# Online GRPO 验证记录（2026-10-12，Asia/Shanghai）

当前入口为 `python -m grpo.train`，配置见 `configs/online.full.json` 和小模型 `configs/online.smoke.json`。下面旧三阶段记录仅保留为历史参考。

## 本次自动化验收

`uv run pytest -q`：**43 passed，1 skipped**。单 GPU 无法执行 NCCL 双 GPU 测试；真实双进程 Gloo 测试通过。

- 连续三个 Online Step，各一次 AdamW 更新；下一轮 Rollout State 与上一轮 checkpoint State 一致。
- P×G 全部轨迹参与累积；microbatch=1 与整批的 State/AdamW moment 等价。
- 冻结 Backbone/BF16 参数，FP32 Initial State 梯度和更新非零；保留 CUDA 对照、终态反传、Activation Checkpointing 与完整 BPTT 原测试。
- 双进程实际在线引擎执行完整组切分、空槽同步、尾批、两个 Epoch 和暂停恢复，更新与单 Rank 等价；真实样本不重复。
- Rollout/Judge/Optimizer 后/Checkpoint 提交前/日志失败的恢复测试，均与连续训练最终 State 和 AdamW moment 一致。
- Python/Torch RNG 恢复；配置/State/待处理 Policy Version 校验；对齐失败不调用 Optimizer，不提交新 Step。
- 逐样本 JSONL 与 SwanLab Text 重组完全一致；超过 SDK 的 108 媒体项上限时分列，避免截断。

## 实际 BF16 模型 + 远程 Judge

硬件：单张 RTX PRO 6000 Blackwell，97,887 MiB；模型 `/mnt/sda1/rwkv_weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth`；复用现有 CUDA kernel。Judge 实际模型 `Qwen3.6-35B-A3B-UD-Q8_K_XL`，8 slots，每 slot context 32768。

本地运行目录 `runs/online-real-validation/train-short`：6 个短真实 Prompt，P=2/G=8/microbatch=2/max_new_tokens=12，LR=1e-4。Completion 全部由模型实际采样，评分全部来自远程 Judge。

| Step | State 更新 Norm | Grad Norm | Ratio Mean | Clip Fraction | 最大 LogProb 误差 | Step 秒（当时阶段计时） |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.075940 | 8.076583 | 1.001725 | 0.005291 | 0.202239 | 8.36 |
| 2 | 0.053773 | 5.807243 | 0.997278 | 0 | 0.140561 | 8.64 |
| 3 | 0.042894 | 17.352558 | 1.002669 | 0.005848 | 0.192908 | 8.42 |

最终 Global Step=3、Policy Version=3、Epoch=1、Sampler cursor=0。再次以 resume=true 启动该已完成运行，Global Step 保持 3，没有执行新训练或新增轨迹。逐条验证第二、三轮 State 指纹等于前轮保存的 State。48 条完整轨迹及三个 Step 指标写入本地 JSONL/SwanLab；全部 13 个 SwanLab 原生 Text 文件内容逐块与原始 JSONL 比对一致。峰值 allocated 596,244,992 bytes。此后计时代码补充了 checkpoint 耗时，最终 Step Time 覆盖原子提交，上传异步处理时间不计入。

## 对齐保护与恢复的实际限制

另一轮 max_new_tokens=48/LR=1e-3 的实验：首轮完成，第二轮 BF16 对齐误差约 0.47、Clip Fraction 约 4%，超过配置阈值后停止。逐条 teacher forcing 和 microbatch=2/8 的复算均有误差，未通过放宽阈值继续更新。

真实暂停实验在 Step 1 后退出，再从 checkpoint 恢复。恢复后的新 Rollout State 指纹与已提交 checkpoint 一致；下一轮被 BF16 对齐门限拦截（包括一个 max_error=0.300430 的短生成实验）。已验证 latest.pt 保持 Step/Policy Version=1，State 与原 Step 1 快照完全一致，没有重复提交 Step 1。自动化测试中的完整恢复到三轮已经通过；真实中途恢复实验尚未完成余下两轮更新。不能将此描述为长序列或任意配置均已通过。

本机仅验证 SwanLab offline 原生 SDK 日志；环境没有 SWANLAB_API_KEY，未验证云端实时上传。生产配置默认 online，需配置用户的 SwanLab 登录；离线日志可用 SDK 提示的 `swanlab sync` 命令上传。NCCL 双卡和 7.2B 连续在线训练未在本次验收运行。

---

## 历史离线验证记录

# 验证记录（2026-10-11，Asia/Shanghai）

当前接口已改为三个独立阶段，命令和数据格式以 README 为准；下面早期记录中的 Stage 2 是当时的训练阶段。

已按用户最终确认的**单卡范围**完成真实工程验证。多卡代码保留；当前没有第二张 GPU，NCCL 双卡实机验证未执行。

## 环境

- NVIDIA RTX PRO 6000 Blackwell Workstation Edition，1 张，约 96 GB 显存。
- 当前目录 `.venv`，`uv.lock` 已生成；PyTorch 2.14.1+cu130，CUDA Toolkit 13.4。
- OpenAI client 3.28.0，SwanLab 0.10.2。
- Backbone：`/mnt/sda1/rwkv_weights/rwkv7-g1d-0.1b-20260129-ctx8192.pth`。
- State_v0：显式零初始化，12 层 × 12 heads × 64 × 64，FP32。
- Judge：实际查询 `/v1/models` 确认 `Qwen3.6-35B-A3B-UD-Q8_K_XL`；成功使用 `/v1/chat/completions` 和 `response_format=json_object`。

## 测试结果

`uv run pytest -q`：**16 passed，1 skipped**。

| 检查 | 实际验证 |
|---|---|
| CUDA Forward / Backward | T=1、7、16、19、32；B=2、H=2；与 FP32 PyTorch 递推比对输出、终态、Initial State 及 r/w/k/v/a/b 全部梯度 |
| 终态反传 | 显式非零 dsT；21 Token 按 7+14 分块，与整段梯度比对 |
| CUDA Stream | 在非默认 Stream 执行，等待后验证数值一致 |
| 组内 Advantage | 不同 Reward 偏移组分别归一化；零方差产生全零；非法 Reward 拒绝 |
| PPO / Mask / EOS | 正负 Advantage 与 clipping；首 Completion Shift；EOS 有效；Padding 无损失；inactive exp 不溢出 |
| 真实模型计算图 | Completion loss 穿过 Prompt / 多个 chunk，各层 Initial State 梯度非零且有限 |
| 冻结 / 更新 | Backbone BF16、所有梯度为空，优化前后所有权重逐元素不变；唯一 FP32 State Parameter 得到梯度并更新 |
| 轨迹隔离 | 同 seed 两次独立 rollout 完全相同，Initial State 未被原地改写 |
| 失败 / 恢复 | 模拟一个 Judge 请求失败；不生成不完整组，不赋零；重试不重复 Rollout 或成功评分；完成后重跑无额外请求 |
| DDP 数学 | 真实两个 Gloo 进程；不等 Token 数、2 次梯度累积、末尾空 Rank；梯度及 SGD 更新与单进程合并 batch 一致 |
| NCCL 双 GPU | 自动测试脚本已提供；当前 1 GPU，明确 skipped |

CUDA 基于用户修正后的 sT/dsT 接口继续实现非整块长度、尾块 checkpoint、当前 Stream 和 kernel launch error 检查，复用原有前向与 backstepping 核心。

## 真实双阶段闭环

两轮均为 P=2、G=3，最长生成 32 Token，共 190 个有效 Completion Token/轮。使用原始数据中的 `000009-001` 和 `000028-001`，Prompt 长度分别为 732 和 728；通过配置的最大 Prompt 长度过滤选出，未截断原文。

| 轮次 | 两个组的真实 Judge Reward | 训练结果 |
|---|---|---|
| v0 → v1 | [0.25, 0, 0.125]；[0, 0.125, 0] | 2 个 optimizer step，保存 `runs/train_v1/latest.pt` |
| v1 → v2 | [0, 0.0625, 0.25]；[0, 0.125, 0.125] | 第 1 步退出，独立进程恢复第 2 步，保存 `runs/train_v2/latest.pt` |

第一轮两个 step 的 State gradient norm 为 3.08425、2.69775；State update norm 为 0.0752970、0.0544436。第二轮分别为 5.16435、4.11657 和 0.0753147、0.0544228。新 State 已实际被下一轮 Stage 1 加载使用。

这些短文本、低参数量 smoke run 验证的是工程闭环和梯度正确性，不是任务质量改善实验；未据此宣称 reward 提升。

## 全部采样 Token 的 old/new 对齐

在同一初始策略下，对每轮全部 6 条轨迹重新 teacher-force：

| 指标 | v0 snapshot | v1 snapshot |
|---|---:|---:|
| 有效 Token 数 | 190 | 190 |
| 最大绝对 LogProb 差 | 0.097578 | 0.097786 |
| 平均绝对 LogProb 差 | 0.019494 | 0.019608 |
| 平均 Ratio | 0.998091 | 0.998297 |
| Ratio 范围 | [0.907032, 1.083973] | [0.906843, 1.093110] |

差异来自 BF16 单步 / 序列矩阵计算形状及舍入；概率定义一致，没有温度、Top-P 或 token shift 的定义差异。测得最大差小于当前默认 0.15 阈值。明细保存在 `runs/alignment_all.json`。

## 恢复的严格对照

为验证恢复，额外对隔离的同一批数据副本运行连续训练 oracle，仅用于数值对照，不产生新的训练轮次：

- 恢复后 State 与连续训练 **逐元素完全相同**，最大差 **0.0**。
- AdamW `exp_avg`、`exp_avg_sq`、`step` 均逐元素完全相同。
- cursor=6，step=2，complete=true。
- 记录：`runs/resume_validation.json`。

重新执行已完成 Stage 1 返回 `Dataset already complete`；重新执行 Stage 2 resume 返回 `Training already complete`，没有追加采样、Judge 请求或 optimizer step。

## 产物

- `runs/state_v0.pt`：可重复使用的零初始化起点。
- `runs/rollout_v0/`、`runs/rollout_v1/`：两个完整离线 Rollout Dataset。
- `runs/train_v1/latest.pt`、`runs/train_v2/latest.pt`：包含优化器状态的 State-only Checkpoint，不包含 Backbone。
- `runs/train_v2/metrics.jsonl`：跨中断恢复的两个 step 指标。
- `runs/train_v2/swanlog/`：已验证写入的 SwanLab offline 文件；恢复对应后续离线 run，训练 step 连续。
- `tests/test_ddp.py`：有两张 GPU 时执行真实单卡 / NCCL 双卡训练及 optimizer moment 对照。

下一轮可以 `runs/train_v2/latest.pt` 作为 Stage 1 / Stage 2 共同初始 State，并使用新的 Dataset 和训练输出目录。

## 追加：全量 G=8 / Judge context 修复

实际 `/props` 确认服务 `192.168.110.25:8000` 是 16 slots × 2048 tokens，位于不同于当前 `.77` 的机器。新增服务容量预检、不可重试 HTTP context 错误的快速停止，以及未消费数据的 G 扩展。实际全量目录已从 G=4 扩大到 G=8；31 个 sample 和 6 个成功评分的文件 SHA256 在迁移前后完全一致。

追加测试覆盖：真实错误格式 `exceed_context_size_error` 仅请求一次；检查每 slot 而非总 context；G=4→8 保存缓存、清除旧归一化结果、禁止扩展已消费数据。当前完整测试为 19 passed、1 skipped（双 GPU 硬件测试）。远端 context 扩容必须通过该机器的 llama-server 启动参数重启完成；本地预检已验证会阻止旧的 2048 配置。

## 追加：G1k 7.2B 权重验证

当前全量配置已切换至 `/mnt/nvme0n1/rwkv7-g1k-7.2b-20260930-ctx25600.pth`，32 层、4096 hidden、64 heads，默认 FP32 零 State `[32,64,64,64]`。单卡真实测试：

- 1031 Token 的真实 Prompt + 8 个实际采样 Token，old/new 最大 LogProb 差 0.069238，平均 Ratio 1.000768。
- State gradient norm 5.815814，AdamW 更新范数 0.280169；Backbone 全部冻结、梯度为空。
- 数据最长 Prompt（6921 Token）加 1024 个合成 Completion Token，仅用于最大长度显存/反向验证：Forward + Backward 成功，峰值 PyTorch allocated 显存 77.919 GiB，State 梯度有限且非零。该合成样本未导出为训练数据。
- Judge 预检已通过：8 slots，每 slot 32768 Token。
- 明细：`runs/g1k_7b_validation.json`。未启动全量长任务；启动命令仍使用两份 `*.full.json`，新模型的数据和 checkpoint 采用独立目录。

## 追加：32 / 64 路并行 Rollout 与 tqdm

- `grpo/rollout.py` 在 GPU 上同时 Prefill 不同 Prompt，再复制独立 WKV / shift state 为 G 条轨迹；每步 Decode 一次处理所有未结束轨迹，已验证模型实际输入含 `[32,1]` / `[64,1]`。
- 长短 Prompt 共批时保留训练的固定 chunk 网格，短尾单独处理；完成 EOS / stop / length 的行立即保存并退出活跃 Batch。Stage 1 使用单行 tqdm，显示采样、评分及完整组进度。
- 增加 CPU 确定性递推对照、长短序列/批次重排/独立 seed/提前结束/中断保存测试，以及真实 GPU 32、64 路 old/new 对齐测试。

以下测试采用 7.2B 权重、最长的 4 / 8 个真实 Prompt（最长 6921 Token）、G=8、生成上限 32 Token：

| 指标 | 4×8=32 | 8×8=64 |
|---|---:|---:|
| 实际生成 Completion Token 数 | 963 | 1980 |
| Rollout 秒数（不含加载与 Judge） | 2.704 | 3.748 |
| 峰值 PyTorch allocated GiB | 16.55 | 19.62 |
| 抽检有效 Token 数 | 141 | 273 |
| old/new 最大绝对 LogProb 差 | 0.176242 | 0.156509 |
| 平均 Ratio | 1.000256 | 0.999455 |
| 初始 Clip Fraction | 0 | 0 |

Initial State 未改写、梯度为空；明细保存在 `runs/g1k_7b_batch_validation.json`。这是单次测量，不作为吞吐保证。

使用原数据前 4 个 Prompt、G=8、生成上限 32，完成了新的独立 Stage 1 → 真实 Judge → Dataset → Stage 2 闭环：

- `runs/batch32_smoke_rollout` 包含 4 个完整组、32 条成功评分。
- `runs/batch32_smoke_train/latest.pt`：4 次优化、cursor=32、complete=true；FP32 State 范数 0.675052，Checkpoint 不含 Backbone。每步执行 Backbone 无梯度、State 梯度有限检查；SwanLab offline 和 metrics.jsonl 正常保存。
- 初始 8 条抽检最大 LogProb 差 0.201777，平均 Ratio 1.000351，Clip Fraction 0.004032（约 0.40%）。
- BF16 的时间/Batch GEMM 形状变化会产生舍入差异。7.2B 全量配置最大绝对差阈值设为 0.25，另加平均 Ratio 偏差 ≤0.02 和初始 Clip Fraction ≤1% 的检查；超限仍拒绝训练。没有把 old logprob 替换成 teacher-forcing 分数来隐藏误差。
- 明细：`runs/batch32_e2e_validation.json`；本次测试配置为 `runs/batch32_stage1.json` 和 `runs/batch32_stage2.json`。

另完成 64 路 × 1024 Token 满长度压力测试：8 个最长真实 Prompt，关闭 EOS / stop 提前结束（仅用于测试，不导出为训练数据），共生成 65536 Token，耗时 28.03 秒（不含加载），峰值 PyTorch allocated **19.62 GiB**。全程无梯度，Initial State 未被改写。结果：`runs/g1k_7b_batch64_long_validation.json`。

本轮完整测试：**31 passed，1 skipped**（仍为 NCCL 双卡硬件测试）。

## 当前版本：三个独立阶段、JSONL 和 manifest

用户最终要求的跨阶段流水线/实时跟随/积压队列已移除。当前 Stage 1 仅 GPU Rollout，Stage 2 仅并发 Judge，Stage 3 仅训练。Stage 2 在 Stage 1 不完整时直接拒绝，不等待或跟随生产者。

实际 7.2B 三阶段验证使用 `runs/three_stage_validation/stage{1,2,3}.json`：

- Stage 1：4 个真实 Prompt × G=8，共 32 条 Completion，生成上限 32 Token。唯一 Rollout 数据输出为 `rollouts.jsonl`，含 metadata、Prompt、Completion、complete 记录。没有调用 Judge。
- Stage 2：在 `CUDA_VISIBLE_DEVICES=''` 环境独立运行，8 个并发请求，32 条评分均成功；保存 `judgments.jsonl`，导出 `manifest.json`。Manifest 包含两个 JSONL 的相对路径和 SHA256、策略版本、组内 Reward/Mean/Std/Advantage 与 dataset_id。
- Stage 3：真实 microbatch=16、累积=1（验证配置），每步 16 条。先运行一步至 cursor=16，再从 checkpoint 恢复至 cursor=32、step=2、complete=true。State 梯度范数 0.422015 / 0.279935，更新范数 0.276408 / 0.193915；Backbone 梯度检查通过，State FP32。
- 初始对齐：最大绝对 LogProb 差 0.256069，平均 Ratio 0.999093，Clip Fraction 0.002041（约 0.20%）。7.2B 全量配置最大误差阈值为 0.30，继续要求平均 Ratio 偏差 ≤0.02、初始 Clip Fraction ≤1%，不覆盖或重算替换 Old LogProb。失败预检不再登记数据集消费。
- 新 checkpoint 已实际用于下一次 Stage 1，生成 `next_rollouts.jsonl`，确认 State version 改变。
- 实际产物和汇总：`runs/three_stage_validation/validation.json`、`manifest.json`、`train/latest.pt`、`train/metrics.jsonl`、`train/swanlog/`。

Stage 3 长序列显存测试（同一 7.2B / 同一 TrainingPolicy）：B=16，每条真实最长 Prompt 6921 Token + 1024 个合成 Completion Token，实际输入 `[16,7944]`；全程 Forward/Backward 与 State 更新成功。峰值 allocated **63.34 GiB**，reserved **64.62 GiB**；State gradient norm 0.562129，Backbone 梯度为空。原生 checkpoint 与不重算版本的输出和 Initial State 梯度已做对照。明细 `runs/micro16_long_validation.json`；合成轨迹没有进入训练数据。

当前完整测试：**34 passed，1 skipped**（NCCL 双卡硬件不足）。新增/更新覆盖 JSONL 末行损坏恢复、不重复采样、Stage 2 拒绝未完成输入、真正的 Judge 并发、成功评分即时持久化、失败只重试缺失评分、manifest 文件校验、策略不匹配、三个入口无跨阶段调用、B=16 checkpoint 梯度等价和预检失败不消费数据。
