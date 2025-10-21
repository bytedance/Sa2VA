import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import master_only
from transformers import AutoModel
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert HF to PTH script')
    parser.add_argument('hf_model_path', help='path to HF model directory')
    parser.add_argument(
        '--save-path', type=str, default=None, help='save path for PTH model')
    parser.add_argument(
        '--arch-type', type=str, required=True, choices=['internvl', 'qwen'],
        help='Model architecture type')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # Load HF model
    hf_model = AutoModel.from_pretrained(
        args.hf_model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).eval()
    print(hf_model)

    hf_state_dict = hf_model.state_dict()
    print(f'Loaded HF model with {len(hf_state_dict)} parameters')

    # Convert HF state dict back to original format
    original_state_dict = {}
    if args.arch_type == 'internvl':
        # Reverse mapping for InternVL based models
        # Original mapping in convert_to_hf.py was:
        # name_map = {'mllm.model.': '', '.gamma': '.g_weight'}
        for key in hf_state_dict.keys():
            new_key = copy.deepcopy(key)
            # Apply reverse mapping
            if new_key.startswith('vision_model') or new_key.startswith('language_model') or new_key.startswith('mlp'):
                new_key = 'mllm.model.' + new_key
            
            if '.g_weight' in new_key:
                new_key = new_key.replace('.g_weight', '.gamma')
            original_state_dict[new_key] = hf_state_dict[key]
    elif args.arch_type == 'qwen':
        # Reverse mapping for QwenVL based models
        # Original mapping in convert_to_hf.py was:
        # name_map = {'mllm.': '', '.gamma': '.g_weight'}
        for key in hf_state_dict.keys():
            new_key = copy.deepcopy(key)
            if new_key.startswith('model'):
                new_key = 'mllm.' + new_key

            if '.g_weight' in new_key:
                new_key = new_key.replace('.g_weight', '.gamma')
            original_state_dict[new_key] = hf_state_dict[key]
    else:
        raise ValueError(f"Unsupported architecture type: {args.arch_type}")


    print(f'Converted to original format with {len(original_state_dict)} parameters')

    # Save the converted state dict
    if args.save_path is None:
        args.save_path = args.hf_model_path.rstrip('/') + '_converted.pth'
    elif not args.save_path.endswith('.pth'):
        args.save_path += '.pth'

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Save the state dict in PTH format
    torch.save(original_state_dict, args.save_path)
    print(f"Saved converted PTH model to {args.save_path}")

    # Print some statistics
    total_params = sum(p.numel() for p in original_state_dict.values())
    print(f"Total parameters: {total_params:,}")

if __name__ == '__main__':
    main()