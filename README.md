# Lightning LLM Training Test
LLM硬體負載與吞吐量測試。

顯存使用量、記憶體使用量、實際算力(FLOP/s)與每秒鐘Token處理量。

[![asciicast](https://asciinema.org/a/7UHAhWtFangpe6ahMmFxpGli7.svg)](https://asciinema.org/a/7UHAhWtFangpe6ahMmFxpGli7)

## Docker
```bash
docker run --rm --gpus all p208p2002/llm-training-test -M gpt2 --batch_size 2 --seq_length 512
```
```
MODEL:gpt2
MODEL_SIZE:124M
GPU_COUNT:1
tokens/sec/GPU:2571.891
tokens/sec/total:2571.891
tflop/sec/GPU:6.019
tflop/sec/total:6.019
peak_ram:5324MB
peak_vram_0:4687MB
```

## 使用
```bash
usage: main.py [-h]
               [--ds_strategy {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--seq_length SEQ_LENGTH]
               [--precision {16,32,64,bf-16,16-mixed,bf16-mixed,32-true,64-true}]
               [--model_name {gpt2,gpt2-medium,gpt2-large,gpt2-xl,bigscience/bloom-560m,bigscience/bloom-1b1,bigscience/bloom-1b7,bigscience/bloom-3b,bigscience/bloom-7b1,bigscience/bloom,huggyllama/llama-7b,huggyllama/llama-13b,huggyllama/llama-30b,huggyllama/llama-65b}]

options:
  -h, --help            show this help message and exit
  --ds_strategy {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}, -s {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
  --batch_size BATCH_SIZE, -B BATCH_SIZE
  --seq_length SEQ_LENGTH, -L SEQ_LENGTH
  --precision {16,32,64,bf-16,16-mixed,bf16-mixed,32-true,64-true}, -p {16,32,64,bf-16,16-mixed,bf16-mixed,32-true,64-true}
  --model_name {gpt2,gpt2-medium,gpt2-large,gpt2-xl,bigscience/bloom-560m,bigscience/bloom-1b1,bigscience/bloom-1b7,bigscience/bloom-3b,bigscience/bloom-7b1,bigscience/bloom,huggyllama/llama-7b,huggyllama/llama-13b,huggyllama/llama-30b,huggyllama/llama-65b}, -M {gpt2,gpt2-medium,gpt2-large,gpt2-xl,bigscience/bloom-560m,bigscience/bloom-1b1,bigscience/bloom-1b7,bigscience/bloom-3b,bigscience/bloom-7b1,bigscience/bloom,huggyllama/llama-7b,huggyllama/llama-13b,huggyllama/llama-30b,huggyllama/llama-65b}
```

程式會在執行10個step後自動結束，並將結果紀錄到`report.txt`

## 計算訓練時間
$$D = Token總數$$

$$N = 模型參數量$$

$$C = 6DN$$

$$T_{days} = C/(FLOPS \times 86400 \times 10^{12})$$

## 最佳化模型參數量與資料集大小
參見: [p208p2002/Compute-Optimal-Model-Estimator](https://github.com/p208p2002/Compute-Optimal-Model-Estimator)
> 若已知實際算力，則`utilization=1`，或使用經驗值0.2~0.3。
