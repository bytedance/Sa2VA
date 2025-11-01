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

    name_map = {'mllm.model.': '', '.gamma': '.g_weight'}

    all_state_dict_new = {}
    for key in all_state_dict.keys():
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    # build the hf format model
    from projects.sasasa2va.hf.models.configuration_sasasa2va_chat import SaSaSa2VAChatConfig
    from projects.sasasa2va.hf.models.modeling_sasasa2va_chat import SaSaSa2VAChatModel

    internvl_config = SaSaSa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = internvl_config.to_dict()
    config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sasasa2va_chat.SaSaSa2VAChatConfig',
         'AutoModel': 'modeling_sasasa2va_chat.SaSaSa2VAChatModel',
         'AutoModelForCausalLM': 'modeling_sasasa2va_chat.SaSaSa2VAChatModel'}

    config_dict["llm_config"]["vocab_size"] = len(model.mllm.tokenizer)
    config_dict["template"] = cfg.template
    sasasa2va_hf_config = SaSaSa2VAChatConfig(
        **config_dict
    )
    hf_sasasa2va_model = SaSaSa2VAChatModel(
        sasasa2va_hf_config, vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model,
    )
    hf_sasasa2va_model.load_state_dict(all_state_dict_new)


    if args.save_path is None:
        args.save_path = f"./{os.path.dirname(args.pth_model)}_{iter_str}_hf"

    hf_sasasa2va_model.save_pretrained(args.save_path)
    model.mllm.tokenizer.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    # copy the files
    os.system(f"cp -pr ./projects/sasasa2va/hf/models/* {args.save_path}")

if __name__ == '__main__':
    main()