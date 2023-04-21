from argparse import ArgumentParser

from model.gpt2_xl.get_instance import get_model,get_config,get_tokenizer
# from model.gpt2_xl_lora.get_instance import get_model,get_config,get_tokenizer
# from model.wz_gpt2_xxl.get_instance import get_model,get_config,get_tokenizer
# from model.wz_gpt2_xxl_lora.get_instance import get_model,get_config,get_tokenizer
# from model.bloomz_7b1_mt.get_instance import get_model,get_config,get_tokenizer
# from model.bloomz_7b1_mt_lora.get_instance import get_model,get_config,get_tokenizer
# from model.bloomz_3b.get_instance import get_model,get_config,get_tokenizer
# from model.bloomz_3b_lora.get_instance import get_model,get_config,get_tokenizer

STRATEGY_DEEPSPEED = 'deepspeed'
STRATEGY_COLOSSAL = 'colossal'

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--strategy', '-s', default=STRATEGY_COLOSSAL,choices=[STRATEGY_DEEPSPEED,STRATEGY_COLOSSAL])
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-6)
    parser.add_argument('--batch_size', '-bs', type=int, default=1)
    args = parser.parse_args()
    return args