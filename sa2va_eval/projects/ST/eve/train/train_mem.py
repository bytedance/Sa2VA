# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from eve.train.train import train

# from eve.train.replace_with_flash_attn import flash_attn_replace
# flash_attn_replace()


if __name__ == "__main__":
    train()
