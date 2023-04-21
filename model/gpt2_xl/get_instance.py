from transformers import AutoTokenizer,GPT2LMHeadModel,GPT2Config

MODEL_NAME = "gpt2-xl"

def get_config():
    return GPT2Config.from_pretrained(MODEL_NAME)

def get_model():
    return GPT2LMHeadModel(get_config())

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
