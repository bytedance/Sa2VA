import argparse
import copy
import os.path as osp
import torch
from mmengine.dist import master_only
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict
import os
import re

def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args():
    parser = argparse.ArgumentParser(description='toHF script')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        '--save-path', type=str, default=None, help='save folder name')
    args = parser.parse_args()
    return args

@master_only
def master_print(msg):
    print(msg)

def main():
    args = parse_args()

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    iter_str = os.path.basename(args.pth_model).split('.')[0]

    model._merge_lora()
    model.mllm.model.language_model.modules_to_save = None
    model.mllm.transfer_to_hf = True

    all_state_dict = model.all_state_dict()

    all_state_dict_new = {}

    print(model)

    # build the hf format model
    from projects.sa2va.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.sa2va.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    from projects.sa2va.hf.models_qwen.configuration_sa2va_chat import Sa2VAChatConfigQwen
    from projects.sa2va.hf.models_qwen.modeling_sa2va_qwen import Sa2VAChatModelQwen

    arch_type = cfg.model.get('arch_type', 'internvl')
    print("arch_type:", arch_type)

    if 'qwen' not in arch_type:
        config = Sa2VAChatConfig.from_pretrained(cfg.path)
    else:
        config = Sa2VAChatConfigQwen.from_pretrained(cfg.path)
    
    config_dict = config.to_dict()
    
    if 'qwen' in arch_type:
        config_dict["text_config"]["vocab_size"] = len(model.mllm.tokenizer)
        # Attention: for qwen, we do untie the weights of embed_tokens and lm_head!
        config_dict["text_config"]["tie_word_embeddings"] = False
    else:
        config_dict["llm_config"]["vocab_size"] = len(model.mllm.tokenizer)

    # For qwen, remove the system prompt
    template_str = cfg.template
    if 'qwen' in arch_type:
        print("Qwen model detected. Removing system prompt from Jinja template.")
        system_prompt_pattern = re.compile(
            r"{% if loop\.first and message\['role'] != 'system' %}.*?<\|im_end\|>\s*{% endif %}",
            re.DOTALL
        )
        template_str = system_prompt_pattern.sub('', template_str)

    config_dict["template"] = template_str


    if 'qwen' in arch_type:
        name_map = {'mllm.': '', '.gamma': '.g_weight'}
        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]

        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfigQwen',
         'AutoModel': 'modeling_sa2va_qwen.Sa2VAChatModelQwen',
         'AutoModelForCausalLM': 'modeling_sa2va_qwen.Sa2VAChatModelQwen'}

        sa2va_hf_config = Sa2VAChatConfigQwen(**config_dict)
    else:
        name_map = {'mllm.model.': '', '.gamma': '.g_weight'}

        for key in all_state_dict.keys():
            new_key = copy.deepcopy(key)
            for _text in name_map.keys():
                new_key = new_key.replace(_text, name_map[_text])
            all_state_dict_new[new_key] = all_state_dict[key]
        
        config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfig',
         'AutoModel': 'modeling_sa2va_chat.Sa2VAChatModel',
         'AutoModelForCausalLM': 'modeling_sa2va_chat.Sa2VAChatModel'}
        
        sa2va_hf_config = Sa2VAChatConfig(**config_dict)

    if 'qwen' in arch_type:
        hf_sa2va_model = Sa2VAChatModelQwen(
            sa2va_hf_config, model=model.mllm.model
        )
    else:
        hf_sa2va_model = Sa2VAChatModel(
            sa2va_hf_config, vision_model=model.mllm.model.vision_model,
            language_model=model.mllm.model.language_model,
        )
    
    missing_keys, unexpected_keys = hf_sa2va_model.load_state_dict(all_state_dict_new)

    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_hf"

    hf_sa2va_model.save_pretrained(args.save_path)

    if 'qwen' in config_dict['architectures'][0].lower():
        model.mllm.processor.save_pretrained(args.save_path)
    else:
        model.mllm.tokenizer.save_pretrained(args.save_path)

    if missing_keys:
        master_print(f"Warning: Missing keys: {missing_keys}")
    if unexpected_keys:
        master_print(f"Warning: Unexpected keys: {unexpected_keys}")
    if not missing_keys and not unexpected_keys:
        master_print("All keys matched successfully!")
    
    print(f"Save the hf model into {args.save_path}")

    # copy the files
    if 'qwen' in arch_type:
        os.system(f"cp -pr ./projects/sa2va/hf/models_qwen/* {args.save_path}")
    else:
        os.system(f"cp -pr ./projects/sa2va/hf/models/* {args.save_path}")

if __name__ == '__main__':
    main()