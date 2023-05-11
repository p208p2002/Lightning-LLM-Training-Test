import lightning.pytorch as pl
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
import time
from typing import Any, Optional
from model.get_instance import get_model, get_tokenizer, get_config
import config
import nvidia_smi
import os, psutil


config_args = config.get_args()
if config_args.strategy == "deepspeed":
    from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
else:
    from colossalai.nn.optimizer import CPUAdam as Adam

from prettytable import PrettyTable


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class LLM(pl.LightningModule):
    ram_usage = 0
    vram_0_usage = 0
    device_num = torch.cuda.device_count()
    tokenizer = get_tokenizer()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if config_args.strategy == "deepspeed":
            self._init()

    def configure_sharded_model(self):
        if config_args.strategy == "colossal":
            self._init()

    def _init(self):
        self.config = get_config()
        self.model = get_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.prof = FlopsProfiler(self.model)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        self._start = time.time()
        self.prof.start_profile()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        input_ids = batch[0]
        seq_len = input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        token_count = seq_len * batch_size
        self._end = time.time()

        # flops
        flops = self.prof.get_total_flops()
        latency = self.prof.get_total_duration()
        tflops_gpu = round(flops / latency / (10**12), 3)
        tflops_total = tflops_gpu * self.device_num
        self.log("tflop/sec/total", tflops_total, prog_bar=True)
        self.prof.end_profile()

        # tokens
        step_exec_time = self._end - self._start
        token_gpu = round(token_count / step_exec_time, 3)
        token_total = token_gpu * self.device_num
        self.log("tokens/sec/total", token_total, prog_bar=True)

        report_template = """
MODEL:{MODEL}
MODEL_SIZE:{MODEL_SIZE}
GPU_COUNT:{GPU_COUNT}
tokens/sec/GPU:{token_gpu}
tokens/sec/total:{token_total}
tflop/sec/GPU:{tflops_gpu}
tflop/sec/total:{tflops_total}
peak_ram:{peak_ram}
peak_vram_0:{peak_vram_0}
        """
        if batch_idx == (10 - 1):
            save_prefix = config_args.model_name.replace("/", "_")
            with open(f"{save_prefix}_report.txt", "w", encoding="utf-8") as f:
                # summary
                f.write(
                    report_template.format_map(
                        {
                            "MODEL": config_args.model_name,
                            "MODEL_SIZE": f"{count_parameters(self.model)//(10**6)}M",
                            "GPU_COUNT": self.device_num,
                            "token_gpu": token_gpu,
                            "token_total": token_total,
                            "tflops_gpu": tflops_gpu,
                            "tflops_total": tflops_total,
                            "peak_ram": f"{self.ram_usage}MB",
                            "peak_vram_0": f"{self.vram_0_usage}MB"
                        }
                    )
                    + "\n"
                )

        return super().on_train_batch_end(outputs, batch, batch_idx)

    def training_step(self, batch, batch_idx):
        #
        encodings = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[0],
        }

        output = self.model(**encodings)
        loss = output["loss"]
        self.log("train_loss", loss)

        # ram usage
        ram_usage = psutil.virtual_memory().used // (1024**2)
        if ram_usage > self.ram_usage:
            self.ram_usage = ram_usage

        # gpu mem
        nvidia_smi.nvmlInit()
        
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        vram_0_usage = info.used // (1024**2)
        if vram_0_usage > self.vram_0_usage:
            self.vram_0_usage = vram_0_usage
        nvidia_smi.nvmlShutdown()

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-5)
        return optimizer
