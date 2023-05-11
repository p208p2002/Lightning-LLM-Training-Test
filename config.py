from argparse import ArgumentParser
from transformers import GPT2LMHeadModel,BloomForCausalLM,LlamaForCausalLM

STRATEGY_DEEPSPEED = 'deepspeed'
STRATEGY_COLOSSAL = 'colossal'

support_model = {
    "gpt2":GPT2LMHeadModel,
    "gpt2-medium":GPT2LMHeadModel,
    "gpt2-large":GPT2LMHeadModel,
    "gpt2-xl":GPT2LMHeadModel,
    "bigscience/bloom-560m":BloomForCausalLM,
    "bigscience/bloom-1b1":BloomForCausalLM,
    "bigscience/bloom-1b7":BloomForCausalLM,
    "bigscience/bloom-3b":BloomForCausalLM,
    "bigscience/bloom-7b1":BloomForCausalLM,
    "bigscience/bloom":BloomForCausalLM
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--strategy', '-s', default=STRATEGY_DEEPSPEED,choices=[STRATEGY_DEEPSPEED,STRATEGY_COLOSSAL])
    parser.add_argument('--learning_rate', '-lr', type=float, default=2e-5)
    parser.add_argument('--batch_size', '-B', type=int, default=1)
    parser.add_argument('--seq_length', '-L', type=int, default=512)
    
    support_models = list(support_model.keys())
    parser.add_argument('--model_name', '-M', choices=support_models,default=support_models[0])
    args = parser.parse_args()
    return args