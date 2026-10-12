# RWKV-7 Online State-only GRPO

一个入口持续执行 **Rollout → 远程 Qwen Judge → GRPO/BPTT → State 更新**。每个 Global Step 选 P 个 Prompt，每个生成 G 条 Completion，所有 P×G 轨迹累积后执行一次 AdamW 更新。下轮直接读取同一实例的最新 FP32 Initial WKV State，冻结的 BF16 Backbone / Embedding / Head 不复制、不保存。

```bash
uv sync --locked
uv run python -m grpo.train --config configs/online.full.json
# 中断后恢复原 Run（不修改 JSON 配置）
uv run python -m grpo.train --config configs/online.full.json --resume
# 恢复验证到总 Global Step=3 后退出
uv run python -m grpo.train --config configs/online.full.json --resume --stop-after-steps 3
# 多 GPU：每个 Rank 一张独立 GPU，分别加载一份 Backbone
uv run torchrun --standalone --nproc_per_node=2 -m grpo.train --config configs/online.full.json
# 仅检查 Judge 模型 ID 和服务端 per-slot context
uv run python -m grpo.judge --config configs/online.full.json
```

修改配置中的模型、词表、数据路径、Judge 地址和输出目录。`state` 为空时从 FP32 零 State 开始，也可导入兼容的 State-only checkpoint。旧 Stage 1/2/3 CLI 和跨 Stage manifest 训练入口已移除；历史数据读取辅助函数仍保留用于检查旧产物。旧离线训练 checkpoint 不包含在线 sampler/RNG 元数据，不能作为在线恢复点；可用兼容的完成 State checkpoint 初始化新运行。

## 配置

| 参数 | 含义 |
|---|---|
| `prompts_per_step` | 每轮全局 Prompt 数 P，尾批允许少于 P |
| `group_size` | 每个 Prompt 的 Completion 数 G，至少 2 |
| `microbatch` | 每个 Rank 单次真正的 batched teacher forcing 大小 |
| `epochs` / `max_steps` | 数据遍历次数 / 总 Global Step 上限；max_steps=0 表示无额外上限 |
| `judge_concurrency` | 并发评分数，1～16 |
| `judge_retry_rounds` / `judge_retry_delay` | 默认 3 个评分轮次 / 2 秒起始退避，只重试缺失项 |
| `length_bucketing` | 默认 true，每个 Rank 内按序列总长度分 Microbatch，减少 Padding |
| `checkpoint_interval` | 历史 checkpoint 保存间隔；latest.pt 每轮必须保存 |
| `chunk_size` | BF16 前向/完整 BPTT chunk，必须与 sampling.prefill_chunk 一致 |
| `activation_checkpointing` | 默认开启 non-reentrant activation recomputation |
| `resume` | 使用原输出目录 latest.pt 完整恢复 |
| `stop_after_steps` | 测试用暂停点，不改变恢复配置契约 |
| `swanlab_mode` | 默认 online；offline 可本地验证后用 swanlab sync 上传 |
| `trace_text_bytes` | 旧配置兼容字段，现已不上传样本文本 |
| `distributed_timeout_seconds` | NCCL collective 超时，默认 300 秒 |

P/G 定义一次 State 更新，microbatch 仅决定内部累积轮数。无需另设 gradient_accumulation；它由本轮完整轨迹数自动决定。当前 objective 为 PPO clipped GRPO，KL beta=0，每轮只使用本版本生成的数据一次。全组 Advantage 都为零时消费该批并记录 skipped_update，不调用 Optimizer、不增加 Policy Version。

## 数据与计算语义

输入 JSONL 为 `{"segments": [{"text": "...Assistant:", "train": false}, {"text": " ...", "train": true}]}`。每个 train=true 段的前文逐字拼接为 Prompt，该段作为参考；不使用未来证据、不套新 chat template。超长 Prompt 跳过，不截断。默认以 seed 稳定打乱符合条件的 Prompt，各 Epoch 复用这一顺序，checkpoint 保存 epoch/cursor。

复用现有 CUDA State Passing Autograd、BF16 Forward/Backward 和完整 BPTT，WKV/shift chunk 边界不 detach。Rollout 复用 GPU Batched Prefill 和 Continuous Batched Decode：相同 Prompt 共享 prefill、每条轨迹状态和采样 RNG 独立；保存逐 Token Old LogProb 和有效 Mask。temperature=1、top_p=1；EOS 和停止文本 Token 保持有效。

Judge 使用严格 JSON Schema，要求四个 0～4 的有限数值及 reason，拒绝缺失/额外字段；reason 最多 160 字符，减少输出开销。远端须支持 OpenAI 兼容的 json_schema response_format，本项目 Qwen/llama-server 已实测支持。保留四维评分、有限重试和上下文校验。Reward 为四维总分/16，每个完整 Prompt Group 计算 population std 和 `(reward-mean)/(std+eps)`。失败不赋零分，不丢弃失败轨迹进行部分更新。成功请求即时持久化，只重试缺失评分。单请求重试中，非法输出会添加结构修正反馈，避免 temperature=0 时重复同一个错误响应；截断仍扩大输出预算。默认最多 3 个评分轮次，各轮间有限退避；持续失败最终停止并保留 pending，绝不赋零分或训练不完整组。永久 HTTP/context 配置错误立即拒绝重试。DDP 在等待期间每秒交换状态心跳，避免一批远程请求耗时超过 NCCL collective 超时。

Teacher forcing 将 `(prompt+completion)[:-1]` 对齐 `[1:]`，仅 Completion Token 计入 Loss。每轮检查全部有效 Token 的 old/new 对齐。平均 Ratio 偏差超过 0.02 或 Clip Fraction 超过 0.01 时，使用当前 State 和原 seed 重放同一 Rank 的完整 Rollout Batch，验证 Completion Token、有效 Mask 以及采样 Old LogProb（最大误差须 ≤1e-5）。验证失败或缺少重放验证时拒绝提交更新；验证通过则保留原 PPO Loss 和原始 Old LogProb 正常更新。仅超限轮次额外执行重放，正常轮次没有额外生成开销。非有限数值、State/Step/Policy Version 不匹配仍直接报错。单 Token Decode 和分块 Teacher Forcing 的 BF16 GEMM/递推舍入路径不同，少量 Token 的较大误差不能单独判定策略过期。`alignment_tolerance=0.30` 用于最大误差告警，记录 `alignment/max_error_exceeded`、最大/平均绝对误差和 RMS，并保留所有采样时的 Old LogProb，不用 Teacher Forcing 分数替换。

训练默认在各 Rank 内按总长度排序后组成 Microbatch，减少 Padding；Group 的 Rank 所属不变，每条真实轨迹仍只参与一次更新。开启 Activation Checkpointing 时，仅有效 Completion 位置执行 FP32 vocabulary LogSoftmax，Prompt/Padding 位置仍完整执行 Backbone 并保留 WKV/shift 图，完整 BPTT 不变。长度排序保留全局 Token 分母，但 BF16 批形状/累加顺序会造成浮点差异；如需原分批顺序可设置 `length_bucketing=false`，可在恢复时调整此执行选项。新增 `training/teacher_forcing_tokens` 和 `training/padding_fraction` 记录实际计算量。

DDP 按完整 Prompt Group 分配各 Rank Rollout；Rank 0 汇总并评分，再广播完整训练批。训练沿同一 Group 分配，不重复使用真实样本。所有 Rank 执行相同累积轮数，空槽用零 Mask dummy；每个 loss 乘 `world_size / 全局有效 Completion Token 数`，最后一槽同步并统一裁剪梯度。每轮检查各 Rank State 指纹。加载、前向、评分、保存及日志的 Python 错误通过集体错误传播通知所有 Rank；CUDA/通信致命故障由 torchrun 和 collective 超时终止，恢复时回到最后提交点。

## Checkpoint 与样本追踪

```text
runs/online_7b/
  latest.pt                     # 每轮原子提交；不含 Backbone
  state-00000010.pt             # checkpoint_interval 历史快照
  pending.json                  # 当前尚未提交轮次的 Rollout、成功评分、策略契约
  steps/00000001.jsonl          # 本轮所有完整样本
  steps/00000001.metrics.json   # 本轮真实统计
  logged.json                   # 已提交轮次的日志游标
  swanlog/                      # 同一持久化 Run ID
```

latest.pt 保存 FP32 State、Optimizer、Scheduler=None（当前固定 LR）、Global Step、Epoch、Sampler cursor、每个 Rank 的 Python/Torch/CUDA RNG、Policy Version、State 指纹和配置/数据/词表/模型契约。恢复需要相同配置和 world_size，设置 resume=true；不会重复消费已提交批次。

更新先写逐样本和指标文件，随后原子替换 latest.pt，最后上传日志并清理 pending。若在 Rollout 中断，整轮重新生成，稳定采样 seed 不变；若评分中断，复用成功评分。若在 backward / optimizer / checkpoint 提交前失败，从上一份 State、Optimizer、RNG 恢复并重新执行这一未提交更新。若提交后日志或清理失败，恢复只补日志，不重做更新。未提交的 steps 文件可能已存在，应以 latest.pt.global_step 判断提交状态。

仅 Rank 0 使用 SwanLab，每个 Global Step 只上传 17 个核心标量：Loss、LR、Grad/State Norm、Ratio、Clip Fraction、Approx KL、有效 Token、跳过更新标记；Reward Mean/Std、零方差组比例；平均 Completion Length、Judge 成功率；Epoch、Policy Version、State 更新幅度。Global Step 作为曲线横轴。

SwanLab 不再上传 performance、trace、逐 Prompt Group Reward、Judge 各维度及其他细分曲线。完整 Prompt/Completion、Reward/Advantage、Judge 明细、Old LogProb/Mask/Token、Step/Policy Version 仍保存于本地 `steps/*.jsonl`；完整统计保留在 `steps/*.metrics.json`，便于排查和追溯。恢复沿用原 Run ID 和 checkpoint，无需修改配置；已存在 Run 的历史图表保留，后续只更新核心指标。


## 验证

```bash
uv run pytest -q
# 小模型 + 远程 Judge（修改本地路径/服务地址后运行）
uv run python -m grpo.train --config configs/online.smoke.json
```

测试覆盖连续三轮最新 State、P×G 完整累积/整批等价、冻结参数、逐 Token 对齐、各阶段故障恢复、零 Advantage、本地完整样本记录和精简 SwanLab 指标，以及真实双进程 Gloo 在线循环（包括尾批、Epoch、恢复和单 Rank 更新等价）。本机实际训练记录见 [VALIDATION.md](VALIDATION.md)。NCCL 双 GPU 测试需要两张 GPU，并通过 GRPO_DDP_CONFIG 指定小模型在线配置。
