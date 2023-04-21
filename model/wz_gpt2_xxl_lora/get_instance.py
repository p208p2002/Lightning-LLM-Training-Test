from transformers import AutoTokenizer,GPT2LMHeadModel,GPT2Config
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

MODEL_NAME = "IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese"

def get_config():
    return GPT2Config.from_pretrained(MODEL_NAME)

def get_model():
    model = GPT2LMHeadModel(get_config())
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    model = get_peft_model(model, peft_config)

    return model

def get_tokenizer():
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # add special token if needed
        if tokenizer.pad_token is None:
            print('set pad_token...')
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if tokenizer.sep_token is None:
            print('set sep_token...')
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
        if tokenizer.eos_token is None:
            print('set eos_token...')
            tokenizer.add_special_tokens({'eos_token': '[EOS]'})
        # add token here
        # tokenizer.add_tokens([INST,PROMPT,OUTPUT],special_tokens=True)
    return tokenizer
