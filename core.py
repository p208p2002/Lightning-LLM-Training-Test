import lightning.pytorch as pl
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
import time
from typing import Any, Dict, Optional
from model.get_instance import get_model, get_tokenizer, get_config
import config
config_args = config.get_args()
from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
from prettytable import PrettyTable
from usage_logger import RamUsageLogger, VramUsageLogger
import torch.distributed as dist

REPORT_TEMPLATE = """
:Model
======================================
Model name:         {model_name}
Model size:         {model_size}
Strategy:           {strategy}
Batch size:         {batch_size}
Seq length:         {seq_length}

:Hardware
======================================
Number gpus:        {gpu_count}
--------------------------------------

:Memory
======================================
Peak CPU RAM:       {peak_ram}
Peak GPU RAM:       {peak_vram}
--------------------------------------

:Token
======================================
Tokens/sec/GPU:     {token_gpu}
Tokens/sec/total:   {token_total}
--------------------------------------

:TFLOP
======================================
TFLOP/sec/GPU:      {tflops_gpu}
TFLOP/sec/total:    {tflops_total}
--------------------------------------
"""


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    return total_params


def write_report(**kwargs):
    with open(f"report.txt", "w", encoding="utf-8") as f:
        # summary
        report = REPORT_TEMPLATE.format_map(kwargs)
        f.write(report + "\n")
    return report

class LLM(pl.LightningModule):
    report = ""
    device_num = torch.cuda.device_count()
    tokenizer = get_tokenizer()

    ram_logger = RamUsageLogger()
    vram_logger = VramUsageLogger()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._init()

    def _init(self):
        self.config = get_config()
        self.model = get_model()
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.prof = FlopsProfiler(self.model)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        if dist.get_rank() == 0:
            self._start = time.time()
            self.prof.start_profile()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int) -> None:
        if dist.get_rank() == 0:
            input_ids = batch[0]
            seq_len = input_ids.shape[-1]
            batch_size = input_ids.shape[0]
            token_count = seq_len * batch_size
            self._end = time.time()

            # flops
            flops = self.prof.get_total_flops()
            latency = self.prof.get_total_duration()
            if latency == 0.0:
                latency = 1e-12
            tflops_gpu = round(flops / latency / (10**12), 3)
            tflops_total = tflops_gpu * self.device_num
            self.log("tflop/sec/total", tflops_total, prog_bar=True)
            self.prof.end_profile()

            # tokens
            step_exec_time = self._end - self._start
            token_gpu = round(token_count / step_exec_time, 3)
            token_total = token_gpu * self.device_num
            self.log("tokens/sec/total", token_total, prog_bar=True)

            if batch_idx == (10 - 1):
                self.report = write_report(
                    model_name=config_args.model_name,
                    batch_size=config_args.batch_size,
                    seq_length=config_args.seq_length,
                    strategy=config_args.strategy,
                    model_size=f"{count_parameters(self.model)//(10**6)}M",
                    gpu_count=self.vram_logger.gpu_count,
                    token_gpu=token_gpu,
                    token_total=token_total,
                    tflops_gpu=tflops_gpu,
                    tflops_total=tflops_total,
                    peak_ram=f"{round(self.ram_logger.get()/10**9,1)}GB",
                    peak_vram=", ".join(
                        [str(round(x / 10**9, 1)) + "GB" for x in self.vram_logger.get()]
                    ),
                )

        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def on_fit_end(self) -> None:
        if dist.get_rank() == 0:
            print(self.report)
        return super().on_fit_end()

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
        self.ram_logger.update()

        # gpu mem
        self.vram_logger.update()

        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-5)
        return optimizer
    
