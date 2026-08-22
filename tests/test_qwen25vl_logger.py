import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).parents[1]
    / 'projects'
    / 'samtok'
    / 'models'
    / 'qwen25vl.py'
)


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class Qwen25VLLoggerTest(unittest.TestCase):

    def test_module_defines_transformers_logger(self):
        expected_logger = object()
        logging = fake_module(
            'transformers.utils.logging',
            get_logger=lambda name: expected_logger,
        )
        modules = {
            'mmengine.model': fake_module('mmengine.model', BaseModel=object),
            'mmengine.config': fake_module(
                'mmengine.config', Config=object, ConfigDict=dict),
            'xtuner.registry': fake_module('xtuner.registry', BUILDER=object()),
            'xtuner.model.utils': fake_module(
                'xtuner.model.utils', find_all_linear_names=lambda model: []),
            'peft': fake_module(
                'peft',
                get_peft_model=lambda *args, **kwargs: None,
                prepare_model_for_kbit_training=lambda model: model,
            ),
            'transformers.modeling_outputs': fake_module(
                'transformers.modeling_outputs', BaseModelOutputWithPast=object),
            'transformers.cache_utils': fake_module(
                'transformers.cache_utils', DynamicCache=object),
            'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl': fake_module(
                'transformers.models.qwen2_5_vl.modeling_qwen2_5_vl',
                Qwen2_5_VLModelOutputWithPast=object,
            ),
            'transformers.utils': fake_module(
                'transformers.utils', logging=logging),
        }
        spec = importlib.util.spec_from_file_location('qwen25vl_under_test',
                                                       MODULE_PATH)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)

        self.assertIs(module.logger, expected_logger)


if __name__ == '__main__':
    unittest.main()
