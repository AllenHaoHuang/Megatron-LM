# 350M MoE Ablation Leaderboard

Codebase: https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/apertus2-ablation

Ablating MoE hyperparameters and activation functions. The baseline is taken from [16-aurora-qkn-moe-32e-tk3-sh1.sbatch](../350m-ablation/runs/16-aurora-qkn-moe-32e-tk3-sh1.sbatch)
which uses MoE and Aurora optimizer.

Refer to [activations.py](../../../megatron/core/activations.py) for activation function implementations.

| rank | entry                           | parent       | change                                                             | optimizer | matrix LR | final 10 avg |    sbatch | wandb | commit                                                                                  |
|-----:|---------------------------------|--------------|--------------------------------------------------------------------| --- | ---: |-------------:|---:| --- |-----------------------------------------------------------------------------------------|
|    1 | xssslupr                        | squared relu | replace squared relu with xssslupr                                 | Aurora + QK norm + MoE | 1e-2 |    **2.137** | [5-xssslupr-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/5-xssslupr-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [iubhfv1a](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/iubhfv1a) | [`b0fddbd`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/commit/b0fddbd) |
|    2 | piecewise polynorm              | polynorm     | remove x^3 term for x<0                                            | Aurora + QK norm + MoE | 1e-2 |        2.139 | [8-piecewise-polynorm-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/8-piecewise-polynorm-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [0bjhmn7e](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/0bjhmn7e) | [`b0fddbd`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/commit/b0fddbd) |
|    3 | polynorm                        | squared relu | replace squared relu with polynorm                                 | Aurora + QK norm + MoE | 1e-2 |        2.141 | [7-polynorm-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/7-polynorm-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [hepi7290](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/hepi7290) | [`b0fddbd`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/commit/b0fddbd) |
|    4 | gxssslupr                       | swiglu       | replace swiglu with gxssslupr                                      | Aurora + QK norm + MoE | 1e-2 |        2.145 | [6-gxssslupr-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/6-gxssslupr-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [wqun8ffe](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/wqun8ffe) | [`b0fddbd`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/commit/b0fddbd) |
|    5 | swiglu                          |              | baseline                                                           | Aurora + QK norm + MoE | 1e-2 |        2.147 | [1-swiglu-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/1-swiglu-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [gi23xffn](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/gi23xffn) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    6 | squared relu                    | swiglu       | replace swiglu with squared relu, mlp dim x1.5 to match parameters | Aurora + QK norm + MoE | 1e-2 |        2.149 | [3-squared-relu-aurora-qkn-moe-32e-tk3-sh1.sbatch](aurora-runs/3-squared-relu-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [awtodkw7](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/awtodkw7) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    7 | swiglu + no shared expert       | swiglu       | remove shared expert, top 4 experts to match parameters            | Aurora + QK norm + MoE | 1e-2 |        2.153 | [2-swiglu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch](aurora-runs/2-swiglu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch) | [2jotyi49](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/2jotyi49) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    8 | squared relu + no shared expert | squared relu | remove shared expert, top 4 experts to match parameters            | Aurora + QK norm + MoE | 1e-2 |        2.155 | [4-squared-relu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch](aurora-runs/4-squared-relu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch) | [28leeo9o](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/28leeo9o) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149)|

## Notes on entries

- **`swiglu`**: baseline from [16-aurora-qkn-moe-32e-tk3-sh1.sbatch](../350m-ablation/runs/16-aurora-qkn-moe-32e-tk3-sh1.sbatch)aurora-qkn-moe-32e-tk3-sh1
- **`swiglu + no shared expert`**: experimenting no shared expert suggested in https://arxiv.org/pdf/2605.11689
- **`squared relu`**: has best throughput and marginally worse than swiglu
- **`squared relu + no shared expert`**: experimenting no shared expert suggested in https://arxiv.org/pdf/2605.11689
- **`xssslupr`**: has best perfomance but large mlp/amax will cause issues when scaling up
- **`gxssslupr`**: better than swiglu here and follows similar scaling laws with swiglu as they are both glus. 
but large mlp/amax will cause issues when scaling up
- **`polynorm`**: better than swiglu
- **`piecewise polynorm`**: better than polynorm, larger scale experiments show it pulls ahead more of polynorm