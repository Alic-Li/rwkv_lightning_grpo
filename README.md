# RWKV-7 State-only GRPO：三个独立阶段

原生 PyTorch / DDP。Stage 1 只做 GPU 并行 Rollout；Stage 2 只做并发 Judge，输出 `manifest.json`；Stage 3 只读取 manifest 训练 Initial WKV State。三个阶段严格顺序执行，每个阶段都可单独重启恢复。没有跨阶段后台任务、实时跟随或积压队列。

## 全量运行

```bash
uv sync --locked

# Stage 1：纯 Rollout，写入一个 JSONL，完成后立即退出
uv run python -m grpo.stage1 --config configs/stage1.full.json

# Stage 2：纯 Judge，不加载 RWKV、不使用本地 GPU
uv run python -m grpo.stage2 --config configs/stage2.full.json

# Stage 3：单卡 State-only GRPO
uv run torchrun --standalone --nproc_per_node=1 \
  -m grpo.stage3 --config configs/stage3.full.json
```

全量配置使用 `/mnt/nvme0n1/rwkv7-g1k-7.2b-20260930-ctx25600.pth`：32 层、4096 hidden、64 heads。Stage 1 / Stage 3 的 `state` 为空字符串、null 或省略时，均创建同一 FP32 全零 Initial WKV State。非空路径必须是有效 State-only checkpoint，加载失败不会静默回退。

全量数据有 2051 行、6898 个目标回合，每回合 G=8，共 55184 条轨迹。默认 Rollout 为 **4 个 Prompt × 8 条 Completion = 32 路**；将 `configs/stage1.full.json` 的 `prompts_per_batch` 改为 `8` 即为 64 路。Judge 并发在 Stage 2 配置中单独设为 8。

Stage 3 默认 `microbatch=16`、`gradient_accumulation=8`，即单卡每 128 条轨迹更新一次。若希望每 16 条更新一次，在开始新训练前将累积次数改为 1。microbatch 是一次真正的 `[B,T]` GPU Forward / Backward，不是串行循环 B=1。三个阶段均显示 tqdm 进度，可用 `progress=false` 关闭；完整训练指标写入 SwanLab 和 metrics.jsonl。

## 数据文件

```text
runs/g1k_7b_v0/
  rollouts.jsonl       # Stage 1 唯一数据输出
  judgments.jsonl      # Stage 2 逐条保存评分及失败记录，用于恢复
  manifest.json        # Stage 2 全部评分成功后导出，Stage 3 唯一入口
  training-use.json    # Stage 3 的数据消费记录
runs/g1k_7b_train_v1/
  latest.pt            # State、AdamW、cursor、step、策略版本；无 Backbone
  alignment.json
  metrics.jsonl
  swanlog/
```

`rollouts.jsonl` 是按行追加的日志，记录类型如下：

- `metadata`：首行保存 Backbone / State / 实现的 hash、policy version、数据和词表 hash、采样参数、Prompt ID 列表、G、seed。
- `prompt`：保存原始 Prompt、Prompt Token IDs、当前目标段参考答案及数据行/段位置；每个 Prompt 只保存一次。
- `completion`：保存 Prompt ID、组内 sample index、Completion 文本和 Token IDs、有效 Mask、逐 Token Old LogProb、seed、停止原因、实际 Rollout Batch 大小。
- `complete`：所有轨迹完成后写入的末行标记。

Stage 1 不生成逐样本 JSON 文件，不连接 Judge。每条 Completion 完成后立刻追加并 fsync；恢复时仅补缺失轨迹。进程异常退出留下的未完整末行，只能由持有文件锁的生产阶段修复。修改 P 不影响恢复，修改 Backbone / State / G / 采样参数则必须使用新的 JSONL。

Stage 2 必须读取带完整结束标记的 Rollout JSONL；未完成时直接报错，不等待 Stage 1、不跟随文件增长。评分期间原始 JSONL 保持不变。成功评分追加到 `judgments.jsonl`，失败样本不记零分；重启只重试尚未成功的样本。

全部组评分成功后，`manifest.json` 保存两个 JSONL 的相对路径及 SHA256、策略和 Judge 元数据、每组 Reward / Mean / Std / Advantage，以及完整性标记和 dataset_id。Stage 3 校验这些信息并从两个 JSONL 还原训练样本。两个 JSONL 和 manifest 是一个整体，复制数据时一起复制。

本次文件格式版本为 2；旧版逐样本目录和旧训练入口不再作为当前接口。历史运行产物保留在原目录，新全量数据写入上述新目录。

## Prompt、采样与评分语义

实际训练文件 `v13_datasets_train.jsonl` 的结构为：

```json
{"segments": [{"text": "...Assistant:", "train": false}, {"text": " ...", "train": true}]}
```

每个 `train=true` 段构成一个目标回合，此前所有段逐字拼接为 Prompt，当前段作为 Judge 参考。历史 assistant 内容属于上下文，未来工具结果不进入 Prompt，不套额外 chat template。任务是生成下一 assistant 回合，可能是工具调用，也可能是最终回答；不执行生成的工具调用。

超过 `max_prompt_tokens` 的 Prompt 跳过，不截断证据。当前全量设置 8192，覆盖全部 6898 个目标回合（最长实际 Prompt 为 6921 Token）。每个 Prompt 的 G 条轨迹从同一版本 Initial State 出发。Prompt Prefill 可共享计算，其后 WKV / shift state 为每条轨迹独立复制。短 Prompt 的尾块单独处理，不让 Padding 改写有效 State；已结束轨迹退出活跃 Batch。

Rollout 全程 `torch.no_grad()`，temperature=1、top_p=1、全词表分类采样，使用 FP32 log-softmax 保存实际采样 Token 的 LogProb。EOS=0 和停止文本 `\n\nUser:` 所在 Token 都保留并设为有效，达到长度上限不伪造 EOS。每条轨迹使用独立稳定 seed；改变 Batch 可能因 BF16 舍入改变未保存的生成结果，已保存记录原样复用。

Judge 使用 OpenAI 兼容 API，先检查实际模型 ID 和每 slot 上下文。Rubric 评估语义、指令遵循、证据支撑、当前回合完成度四维，每项 0～4，Reward 为总分 / 16。合法的下一步工具调用可得分，参考不是唯一正确措辞。每个 Prompt 独立计算 population std 和 `A=(R-mean)/(std+eps)`，零方差为全零，不跨 Prompt 归一化。

Judge 并发可设 1～16，当前为 8。请求有超时和有限重试；JSON 因输出长度截断时扩大响应预算，默认 256→512→1024，不修改输入或 Rubric。拒绝非法/缺失/越界评分。确定性 HTTP 4xx（408/409/429 除外）直接报错。不会截断输入以规避 context overflow。

```bash
# 单独检查远端模型和上下文，不执行 Rollout
uv run python -m grpo.judge --config configs/stage2.full.json
```

当前 Judge 地址 `http://192.168.110.25:8000`，模型 `Qwen3.6-35B-A3B-UD-Q8_K_XL`。全量配置要求每 slot 至少 16384 Token；实际验证服务器为 8 slots × 32768 Token。以 `/props` 为准，API 无法扩大已分配的服务端 context。

## Stage 3：训练计算图与恢复

RWKV BF16 方程来自 `/home/alic-li/RWKV-LM-V7`，直接复用本项目 `cuda/` 的 State Passing Forward / Backward。Backbone、Embedding、Head 全部冻结，只有 `[layer,head,value,key]` 的 FP32 Initial State 是可训练 Parameter。

Teacher forcing 使用 `(prompt+completion)[:-1]` 预测 `[1:]`，Completion 第一 Token 对应 Prompt 最后位置的 logits。Prompt / Padding mask=0，实际 Completion（含 EOS）mask=1。采用标准 PPO clipped surrogate：`ratio=exp(new-old)`，正负 Advantage 均参与，按整个优化窗口的有效 Completion Token 数归一化。当前 K=1、KL beta=0。

默认开启 `activation_checkpointing=true`。`grpo/training.py` 使用原生 non-reentrant checkpoint，重算每个时间 chunk 的激活。WKV / shift 边界始终保留计算图，Prompt 不 detach，不做截断 BPTT。该实现降低 B=16 的激活显存，但增加重算开销。

DDP 将 microbatch loss 乘以 `world_size / 全局窗口有效 Token 数`，前几次累积使用 `no_sync()`，最后一次同步。尾部只执行必要的累积轮次，没有数据的 Rank 用零 mask dummy 同步。梯度裁剪在累积/同步后进行。只有 Rank 0 写 tqdm、SwanLab、指标和 checkpoint。

```bash
# 有两张独立 GPU 时
uv run torchrun --standalone --nproc_per_node=2 \
  -m grpo.stage3 --config configs/stage3.full.json
```

训练启动按实际 microbatch 重算 old/new 对齐。7.2B 配置抽检 16 条，最大 LogProb 误差上限 0.30，同时要求平均 Ratio 与 1 偏差 ≤0.02、初始 Clip Fraction ≤1%；非有限数值或超限拒绝训练。BF16 的单步/序列 GEMM 尺寸差异会造成舍入误差，不应通过放宽阈值掩盖 Shift、Mask 或策略版本错误。

`resume=true` 从 checkpoint 的 State / AdamW / cursor / step 恢复，需保持原训练配置。完成的数据集不能被静默再次消费。下一轮将已完成的 `latest.pt` 同时设为 Stage 1 和 Stage 3 的初始 State，并使用新的 JSONL、manifest 和训练输出目录；禁止把新 State 配给旧策略数据。

## 验证

```bash
uv run pytest -q
uv run torchrun --standalone --nproc_per_node=2 -m tests.ddp_worker
```

已实测本机 7.2B、64 路各生成满 1024 Token，Rollout 峰值 allocated 19.62 GiB。Stage 3 B=16、每条最长 Prompt 6921 Token + 1024 个合成 Completion Token，完整 Forward/Backward 峰值 63.34 GiB，State 梯度和更新非零，Backbone 梯度为空。合成样本仅用于压力测试，不进入训练数据。

单元测试包括 CUDA 对照/终态反传/Stream、EOS/Padding/Mask、32/64 路采样、梯度检查点与直接反传等价、JSONL 中断恢复、并发 Judge、完整组 Advantage、文件完整性、策略拒绝及两个 Gloo 进程归一化。仅有一张 GPU，因此 NCCL 双卡测试明确 skip。实际三阶段验证及历史记录见 [VALIDATION.md](VALIDATION.md)。
