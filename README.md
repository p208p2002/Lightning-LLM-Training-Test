# Lightning LLM Training Test
LLM硬體負載與吞吐量測試。

顯存使用量、記憶體使用量、實際算力(FLOP/s)與每秒鐘Token處理量。

## 輸出示例
以下的結果使用 Tesla T4*4

```
:Model
======================================
Model name:         gpt2
Model size:         124M
Strategy:           deepspeed_stage_2_offload
Batch size:         1
Seq length:         512

:Hardware
======================================
Number gpus:        4
--------------------------------------

:Memory
======================================
Peak CPU RAM:       12.7GB
Peak GPU RAM:       3.8GB, 3.8GB, 3.6GB, 3.6GB
--------------------------------------

:Token
======================================
Tokens/sec/GPU:     1587.348
Tokens/sec/total:   6349.392
--------------------------------------

:TFLOP
======================================
TFLOP/sec/GPU:      3.795
TFLOP/sec/total:    15.18
--------------------------------------
```


## 使用
### Docker
```bash
docker run --shm-size 8G --rm --gpus all p208p2002/llm-training-test --model_name gpt2 --batch_size 2 --seq_length 512
```

### Python
```bash
usage: main.py [-h]
               [--strategy {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}]
               [--learning_rate LEARNING_RATE] [--batch_size BATCH_SIZE]
               [--seq_length SEQ_LENGTH]
               [--precision {16,32,64,bf-16,16-mixed,bf16-mixed,32-true,64-true}]
               [--model_name {gpt2,gpt2-medium,gpt2-large,gpt2-xl,bigscience/bloom-560m,bigscience/bloom-1b1,bigscience/bloom-1b7,bigscience/bloom-3b,bigscience/bloom-7b1,bigscience/bloom,huggyllama/llama-7b,huggyllama/llama-13b,huggyllama/llama-30b,huggyllama/llama-65b}]

options:
  -h, --help            show this help message and exit
  --strategy {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}, -s {deepspeed_stage_1,deepspeed_stage_2,deepspeed_stage_3,deepspeed_stage_1_offload,deepspeed_stage_2_offload,deepspeed_stage_3_offload}
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
