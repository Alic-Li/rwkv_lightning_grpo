# RWKV-7 Online State-only GRPO

一个入口持续执行 **Rollout → 远程 Qwen Judge → GRPO/BPTT → State 更新**。每个 Global Step 选 P 个 Prompt，每个生成 G 条 Completion，所有 P×G 轨迹累积后执行一次 AdamW 更新。下轮直接读取同一实例的最新 FP32 Initial WKV State，冻结的 BF16 Backbone / Embedding / Head 不复制、不保存。

```bash
uv sync --locked
uv run python -m grpo.train --config configs/online.full.json
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
| `checkpoint_interval` | 历史 checkpoint 保存间隔；latest.pt 每轮必须保存 |
| `chunk_size` | BF16 前向/完整 BPTT chunk，必须与 sampling.prefill_chunk 一致 |
| `activation_checkpointing` | 默认开启 non-reentrant activation recomputation |
| `resume` | 使用原输出目录 latest.pt 完整恢复 |
| `stop_after_steps` | 测试用暂停点，不改变恢复配置契约 |
| `swanlab_mode` | 默认 online；offline 可本地验证后用 swanlab sync 上传 |
| `trace_text_bytes` | 每个 SwanLab Text 的保守 UTF-8 分块上限，完整数据不截断 |
| `distributed_timeout_seconds` | NCCL collective 超时，默认 300 秒 |

P/G 定义一次 State 更新，microbatch 仅决定内部累积轮数。无需另设 gradient_accumulation；它由本轮完整轨迹数自动决定。当前 objective 为 PPO clipped GRPO，KL beta=0，每轮只使用本版本生成的数据一次。全组 Advantage 都为零时消费该批并记录 skipped_update，不调用 Optimizer、不增加 Policy Version。

## 数据与计算语义

输入 JSONL 为 `{"segments": [{"text": "...Assistant:", "train": false}, {"text": " ...", "train": true}]}`。每个 train=true 段的前文逐字拼接为 Prompt，该段作为参考；不使用未来证据、不套新 chat template。超长 Prompt 跳过，不截断。默认以 seed 稳定打乱符合条件的 Prompt，各 Epoch 复用这一顺序，checkpoint 保存 epoch/cursor。

复用现有 CUDA State Passing Autograd、BF16 Forward/Backward 和完整 BPTT，WKV/shift chunk 边界不 detach。Rollout 复用 GPU Batched Prefill 和 Continuous Batched Decode：相同 Prompt 共享 prefill、每条轨迹状态和采样 RNG 独立；保存逐 Token Old LogProb 和有效 Mask。temperature=1、top_p=1；EOS 和停止文本 Token 保持有效。

Judge 使用现有严格四维评分、有限重试和上下文校验。Reward 为四维总分/16，每个完整 Prompt Group 计算 population std 和 `(reward-mean)/(std+eps)`。失败不赋零分，不丢弃失败轨迹进行部分更新。成功请求即时持久化，下次只重试缺失评分。

Teacher forcing 将 `(prompt+completion)[:-1]` 对齐 `[1:]`，仅 Completion Token 计入 Loss。每轮检查全部有效 Token 的 old/new 对齐；最大误差、平均 Ratio 和 Clip Fraction 超限拒绝提交更新，默认阈值为 0.30 / 0.02 / 0.01。阈值用于 BF16 批尺寸舍入差异，不能用于掩盖策略混用。

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

仅 Rank 0 使用 SwanLab。每个 Global Step 上传 Loss、LR、Grad/State Norm、Ratio、Clip Fraction、Approx KL、有效 Token；Reward/Advantage 分布及逐组 Reward、零方差比例；生成长度/停止原因/吞吐；Judge 四维分数、成功/失败率、请求级重试错误与评分延迟；各阶段/总耗时、端到端 tok/s、各 Rank 最大峰值显存；Step/Epoch/版本/更新幅度。

每轮完整 JSONL 同时作为分块 SwanLab Text 上传，包含 Prompt、Completion、Reward、Advantage、Judge 原始评分/理由/重试记录、Token IDs、Mask、Old LogProb、seed、Policy Version 和 Step ID，可通过 `trace/samples` 追溯。使用 SDK 自带异步上传，无额外 Worker 或队列；本地 JSONL 始终为完整来源。Text 序列化和本地 checkpoint 有同步开销，规模较大时可调整 P/G，上传离线数据用 `swanlab sync <run-directory>`。网络确认与日志游标不是原子事务，提交后崩溃可能重传同一步日志，但不会重做 Optimizer Step。

## 验证

```bash
uv run pytest -q
```

测试覆盖连续三轮最新 State、P×G 完整累积/整批等价、冻结参数、逐 Token 对齐、各阶段故障恢复、零 Advantage、完整 SwanLab 样本记录，以及真实双进程 Gloo 在线循环（包括尾批、Epoch、恢复和单 Rank 更新等价）。本机实际训练记录见 [VALIDATION.md](VALIDATION.md)。NCCL 双 GPU 测试需要两张 GPU，并通过 GRPO_DDP_CONFIG 指定小模型在线配置。
