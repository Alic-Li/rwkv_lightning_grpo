RWKV model equations and `grpo/tokenizer.py` derive from BlinkDL/RWKV-LM,
local reference `/home/alic-li/RWKV-LM-V7` (Apache-2.0, see LICENSE-RWKV).
The State Passing CUDA files were provided in this project; changes preserve
its recurrence and backstepping implementation, adding arbitrary sequence
lengths, current CUDA stream launch, and launch error checking.
