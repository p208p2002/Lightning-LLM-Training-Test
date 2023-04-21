from torch.utils.data import Dataset
import json
from .config import INST,PROMPT,OUTPUT
from config import get_tokenizer,get_config

MODEL_CONFIG = get_config()

class AlpacaDataset(Dataset):
    def __init__(self):
        self.tokenizer = get_tokenizer()
        self.data = json.loads(open("data/alpaca_data.json").read())
    
    def count_tokens(self,ignore_pad_token=True)->int:
        total = 0
        for input_ids,_ in self:
            if ignore_pad_token:
                total += (input_ids != self.tokenizer.pad_token_id).sum().item()
            else:
                total += input_ids.shape[-1]
        return total

    def __getitem__(self,index):
        data = self.data[index]
        instruction = data["instruction"]
        prompt = data["input"]
        output = data["input"]
        model_input_text = f"{INST}{instruction}\n{PROMPT}{prompt}\n{OUTPUT}{output}{self.tokenizer.sep_token}"
        model_input = self.tokenizer(model_input_text,padding='max_length',max_length=MODEL_CONFIG.n_positions,truncation='longest_first',return_tensors="pt")
        
        input_ids = model_input["input_ids"][0]
        attention_mask = model_input["attention_mask"][0]

        return input_ids,attention_mask
    
    def __len__(self):
        return len(self.data)