import abc
import psutil
import nvidia_smi
import torch

class BaseUsageLogger(abc.ABC):
    @abc.abstractclassmethod
    def update(self):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def get(self):
        raise NotImplementedError


class RamUsageLogger(BaseUsageLogger):
    """
    Log RAM usage in bytes
    """
    ram_usage = 0
    
    def update(self):
        ram_usage = psutil.virtual_memory().used
        if ram_usage > self.ram_usage:
            self.ram_usage = ram_usage
        return None
    
    def get(self):
        return self.ram_usage

class VramUsageLogger(BaseUsageLogger):
    """
    Log VRAM usage in bytes
    """
    gpu_count = torch.cuda.device_count()
    vram_usages = [0]*gpu_count
    
    
    def update(self):
        nvidia_smi.nvmlInit()
        
        for gpu_idx,vram_usage in enumerate(self.vram_usages):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_idx)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            
            if vram_usage < info.used:
                self.vram_usages[gpu_idx] = info.used
            
        nvidia_smi.nvmlShutdown()
        return None
    
    def get(self):
        return self.vram_usages
    
        
    
