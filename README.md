# Lightning LLM Training Test
消費級硬體LLM訓練測試

## 訓練與優化策略
- [DeepSpeed](https://www.deepspeed.ai/)
- [Colossal AI](https://colossalai.org/)
- [LoRA(PEFT)](https://github.com/huggingface/peft)

## 硬體環境
|CPU|GPU|RAM|DISK|
|---|---|---|---|
|i7-12700|3090ti(24GB)*1|128GB|12TB Hybrid Drive|

## 重點套件版本
- torch==1.13.1+cu116
- transformers==4.24.0
- deepspeed==0.8.3
- colossalai==0.2.8
- lightning==2.0.1
- lightning-colossalai==0.1.0

## 測試報告
### DeepSpeed 
|全調參             |GPT 1.5B|WZ-GPT 3.5B|BLOOM 3B|BLOOM 7B|
|------------------|:------:|:---------:|:------:|:------:|
|Offload           |✔       |✔          |✔       |CUDA_OOM|
|Offload + Infinity|✔       |✔          |✔       |不穩定   |

|LoRA              |GPT 1.5B|WZ-GPT 3.5B|BLOOM 3B|BLOOM 7B|
|------------------|:------:|:---------:|:------:|:------:|
|Offload           |✔       |✔          |✔       |CUDA_OOM|
|Offload + Infinity|✔       |✔          |✔       |✔       |


### Colossal
|全調參             |GPT 1.5B|WZ-GPT 3.5B|BLOOM 3B|BLOOM 7B|
|------------------|:------:|:---------:|:------:|:------:|
|Default           |✔       |✔          |✔       |RAM_OOM |


|LoRA            |GPT 1.5B|WZ-GPT 3.5B|BLOOM 3B|BLOOM 7B|
|----------------|:------:|:---------:|:------:|:------:|
|Default         |Error     |Error    |Error   |Error   |

## Usage
### Environment Setup
```bash
poetry install --with dev
poetry run poe setup-env
```
### Activate Environment
```bash
poetry shell
```
### Deactivate Environment
```bash
deactivate
```
