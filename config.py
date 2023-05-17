from argparse import ArgumentParser
from transformers import GPT2LMHeadModel, BloomForCausalLM, LlamaForCausalLM

DEEPSPEED_STRATEGY_STAGE_1 = "deepspeed_stage_1"
DEEPSPEED_STRATEGY_STAGE_2 = "deepspeed_stage_2"
DEEPSPEED_STRATEGY_STAGE_3 = "deepspeed_stage_3"
DEEPSPEED_STRATEGY_STAGE_1_OFFLOAD = "deepspeed_stage_1_offload"
DEEPSPEED_STRATEGY_STAGE_2_OFFLOAD = "deepspeed_stage_2_offload"
DEEPSPEED_STRATEGY_STAGE_3_OFFLOAD = "deepspeed_stage_3_offload"

support_model = {
    "gpt2": GPT2LMHeadModel,
    "gpt2-medium": GPT2LMHeadModel,
    "gpt2-large": GPT2LMHeadModel,
    "gpt2-xl": GPT2LMHeadModel,
    "bigscience/bloom-560m": BloomForCausalLM,
    "bigscience/bloom-1b1": BloomForCausalLM,
    "bigscience/bloom-1b7": BloomForCausalLM,
    "bigscience/bloom-3b": BloomForCausalLM,
    "bigscience/bloom-7b1": BloomForCausalLM,
    "bigscience/bloom": BloomForCausalLM,
    "huggyllama/llama-7b":LlamaForCausalLM,
    "huggyllama/llama-13b":LlamaForCausalLM,
    "huggyllama/llama-30b":LlamaForCausalLM,
    "huggyllama/llama-65b":LlamaForCausalLM
}


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--ds_strategy",
        "-s",
        default=DEEPSPEED_STRATEGY_STAGE_2_OFFLOAD,
        choices=[
            DEEPSPEED_STRATEGY_STAGE_1,
            DEEPSPEED_STRATEGY_STAGE_2,
            DEEPSPEED_STRATEGY_STAGE_3,
            DEEPSPEED_STRATEGY_STAGE_1_OFFLOAD,
            DEEPSPEED_STRATEGY_STAGE_2_OFFLOAD,
            DEEPSPEED_STRATEGY_STAGE_3_OFFLOAD,
        ],
    )
    parser.add_argument("--learning_rate", "-lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", "-B", type=int, default=1)
    parser.add_argument("--seq_length", "-L", type=int, default=512)
    parser.add_argument("--precision", "-p", type=str, default="16-mixed",choices=[
        "16","32","64",
        "bf-16",
        "16-mixed", "bf16-mixed", 
        "32-true", "64-true",
    ])

    support_models = list(support_model.keys())
    parser.add_argument(
        "--model_name", "-M", choices=support_models, default=support_models[0]
    )
    args = parser.parse_args()
    return args
