from torch.utils.data import Dataset
from model.get_instance import get_tokenizer,get_config
from config import get_args
import torch

MODEL_CONFIG = get_config()
args = get_args()

class FooDataset(Dataset):
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.num_of_data = 100000
    
    def count_tokens(self,ignore_pad_token=True)->int:
        total = 0
        for input_ids,_ in self:
            if ignore_pad_token:
                total += (input_ids != self.tokenizer.pad_token_id).sum().item()
            else:
                total += input_ids.shape[-1]
        return total

    def __getitem__(self,index):
        
        input_ids = torch.randint(1000,20000,(args.seq_length,))
        attention_mask = torch.ones_like(input_ids)
        return input_ids,attention_mask
    
    def __len__(self):
        return self.num_of_data