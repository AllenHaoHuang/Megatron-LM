# 350M MoE Ablation Leaderboard

Codebase: https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/apertus2-ablation

Ablating MoE hyperparameters and activation functions. The baseline is taken from [16-aurora-qkn-moe-32e-tk3-sh1.sbatch](../350m-ablation/runs/16-aurora-qkn-moe-32e-tk3-sh1.sbatch)
which uses MoE and Aurora optimizer/

| rank | entry                     | change                                                                                                                         | optimizer | matrix LR | final 10 avg |    sbatch | wandb | commit                                                                                  |
|-----:|---------------------------|--------------------------------------------------------------------------------------------------------------------------------| --- | ---: |-------------:|---:| --- |-----------------------------------------------------------------------------------------|
|    1 | swiglu                    |                                                                                                                                | Aurora + QK norm + MoE | 1e-2 |    **2.147** | [1-swiglu-aurora-qkn-moe-32e-tk3-sh1.sbatch](runs/1-swiglu-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [gi23xffn](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/gi23xffn) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    2 | squared relu              | replace swiglu with squared relu, mlp dim x1.5 to match parameters                                                             | Aurora + QK norm + MoE | 1e-2 |        2.149 | [3-squared-relu-aurora-qkn-moe-32e-tk3-sh1.sbatch](runs/3-squared-relu-aurora-qkn-moe-32e-tk3-sh1.sbatch) | [awtodkw7](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/awtodkw7) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    3 | swiglu + no shared expert | remove shared expert, top 4 experts to match parameters                                                                        | Aurora + QK norm + MoE | 1e-2 |        2.153 | [2-swiglu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch](runs/2-swiglu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch) | [2jotyi49](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/2jotyi49) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149) |
|    4 | squared relu + no shared expert | replace swiglu with squared relu, mlp dim x1.5 to match parameters, remove shared expert, top 4 experts to match parameters | Aurora + QK norm + MoE | 1e-2 |        2.155 | [4-squared-relu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch](runs/4-squared-relu-aurora-qkn-moe-no-shared-32e-tk3-sh1.sbatch) | [28leeo9o](https://wandb.ai/saesara/megatron-lm-research-baseline/runs/28leeo9o) | [`c5d4149`](https://github.com/AllenHaoHuang/megatron-lm-research-baseline/tree/c5d4149)|

## Notes on entries

- **`swiglu`**: baseline from [16-aurora-qkn-moe-32e-tk3-sh1.sbatch](../350m-ablation/runs/16-aurora-qkn-moe-32e-tk3-sh1.sbatch)aurora-qkn-moe-32e-tk3-sh1
- **`swiglu + no shared expert`**: experimenting no shared expert suggested in https://arxiv.org/pdf/2605.11689
- **`squared relu`**: replace swiglu with squared relu 
- **`squared relu + no shared expert`**: replace swiglu with squared relu wtih no shared expert

## TLDR
- Shared expert improves performance
- 