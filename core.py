import lightning.pytorch as pl
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler
import time
from typing import Any,Optional
from config import get_model,get_tokenizer,get_config
import config
config_args = config.get_args()
if config_args.strategy == 'deepspeed':
    from deepspeed.ops.adam import DeepSpeedCPUAdam as Adam
else:
    from colossalai.nn.optimizer import CPUAdam as Adam

class LLM(pl.LightningModule):
    avg_tps = 0
    device_num = torch.cuda.device_count()
    tokenizer = get_tokenizer()
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if config_args.strategy == 'deepspeed':
            self._init()
        
    def configure_sharded_model(self):
        if config_args.strategy == 'colossal':
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
        token_count = seq_len*batch_size
        self._end = time.time()
        
        flops = self.prof.get_total_flops()
        latency = self.prof.get_total_duration()
        self.log("TFLOP/s",round(flops/latency/(10**12),3),prog_bar=True)
        # self.prof.print_model_profile()

        self.prof.end_profile()
        
        step_exec_time = self._end - self._start
        token_pre_sec = (token_count/step_exec_time)*self.device_num
        self.log("TOKEN/s",token_pre_sec,prog_bar=True)
        
        # self.avg_tps += (token_pre_sec-self.avg_tps)/(self.global_step+1)
        # self.log("avg_tps",self.avg_tps,prog_bar=True)
        
        return super().on_train_batch_end(outputs, batch, batch_idx)
    
    def training_step(self, batch, batch_idx):
        #
        encodings = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'labels': batch[0]
        }
        
        output = self.model(**encodings)
        loss = output['loss']
        self.log("train_loss", loss)

        return loss
    
    def configure_optimizers(self):        
        optimizer = Adam(self.parameters(),lr=3e-5)
        return optimizer
