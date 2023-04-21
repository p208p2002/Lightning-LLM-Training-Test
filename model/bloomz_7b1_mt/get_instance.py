from transformers import AutoTokenizer,BloomForCausalLM,BloomConfig

MODEL_NAME = "bigscience/bloomz-7b1-mt"

def get_config():
    config = BloomConfig.from_pretrained(MODEL_NAME)
    config.n_positions=128
    return config

def get_model():
    return BloomForCausalLM(get_config())

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
