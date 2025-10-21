
PROMPT_TEMPLATE = dict(
    default=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}\n<|Bot|>:',
        SEP='\n'),
    zephyr=dict(
        SYSTEM='<|system|>\n{system}\n',
        INSTRUCTION='<|user|>\n{input}\n<|assistant|>\n',
        SEP='\n'),
    internlm_chat=dict(
        SYSTEM='<|System|>:{system}\n',
        INSTRUCTION='<|User|>:{input}<eoh>\n<|Bot|>:',
        SUFFIX='<eoa>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<eoa>']),
    internlm2_chat=dict(
        SYSTEM='<|im_start|>system\n{system}<|im_end|>\n',
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>']),
    moss_sft=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<|Human|>: {input}<eoh>\n',
        SEP='\n',
        STOP_WORDS=['<eoc>', '<eom>']),
    llama2_chat=dict(
        SYSTEM=(
            '[INST] <<SYS>>\n You are a helpful, respectful and honest '
            'assistant. Always answer as helpfully as possible, while being '
            'safe. Your answers should not include any harmful, unethical, '
            'racist, sexist, toxic, dangerous, or illegal content. Please '
            'ensure that your responses are socially unbiased and positive in '
            'nature.\n{system}\n<</SYS>>\n [/INST] '),
        INSTRUCTION='[INST] {input} [/INST]',
        SEP='\n'),
    code_llama_chat=dict(
        SYSTEM='{system}\n', INSTRUCTION='[INST] {input} [/INST]'),
    chatglm2=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='[Round {round}]\n\n问：{input}\n\n答：',
        SEP='\n\n'),
    chatglm3=dict(
        SYSTEM='<|system|>\n{system}',
        INSTRUCTION='<|user|>\n{input}<|assistant|>\n',
        SEP='\n'),
    qwen_chat=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    baichuan_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_102>{input}<reserved_103>',
        SEP='\n'),
    baichuan2_chat=dict(
        SYSTEM='{system}\n',
        INSTRUCTION='<reserved_106>{input}<reserved_107>',
        SEP='\n'),
    wizardlm=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    wizardcoder=dict(
        SYSTEM=(
            'Below is an instruction that describes a task. '
            'Write a response that appropriately completes the request.\n\n'
            '{system}\n '),
        INSTRUCTION=('### Instruction:\n{input}\n\n### Response:'),
        SEP='\n\n'),
    vicuna=dict(
        SYSTEM=('A chat between a curious user and an artificial '
                'intelligence assistant. The assistant gives '
                'helpful, detailed, and polite answers to the '
                'user\'s questions. {system}\n '),
        INSTRUCTION=('USER: {input} ASSISTANT:'),
        SEP='\n'),
    deepseek_coder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    # TODO: deprecation, v0.2.0
    deepseekcoder=dict(
        SYSTEM=('You are an AI programming assistant, utilizing '
                'the DeepSeek Coder model, developed by DeepSeek'
                'Company, and you only answer questions related '
                'to computer science. For politically sensitive '
                'questions, security and privacy issues, and '
                'other non-computer science questions, you will '
                'refuse to answer. {system}\n'),
        INSTRUCTION=('### Instruction:\n{input}\n### Response:\n'),
        SEP='\n'),
    deepseek_moe=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    deepseek_v2=dict(
        SYSTEM='{system}\n\n',
        INSTRUCTION='User: {input}\n\nAssistant: ',
        SUFFIX='<｜end▁of▁sentence｜>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<｜end▁of▁sentence｜>']),
    mistral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    mixtral=dict(
        SYSTEM=('[INST] {system} [/INST]\n'),
        INSTRUCTION=('[INST] {input} [/INST]'),
        SEP='\n'),
    minicpm=dict(INSTRUCTION=('<用户> {input} <AI>'), SEP='\n'),
    minicpm3=dict(
        SYSTEM=('<|im_start|>system\n{system}<|im_end|>\n'),
        INSTRUCTION=('<|im_start|>user\n{input}<|im_end|>\n'
                     '<|im_start|>assistant\n'),
        SUFFIX='<|im_end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|im_end|>', '<|endoftext|>']),
    gemma=dict(
        # `system` field is extended by xtuner
        SYSTEM=('<start_of_turn>system\n{system}<end_of_turn>\n'),
        INSTRUCTION=('<start_of_turn>user\n{input}<end_of_turn>\n'
                     '<start_of_turn>model\n'),
        SUFFIX='<end_of_turn>',
        SUFFIX_AS_EOS=False,
        SEP='\n',
        STOP_WORDS=['<end_of_turn>']),
    cohere_chat=dict(
        SYSTEM=('<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system}'
                '<|END_OF_TURN_TOKEN|>'),
        INSTRUCTION=(
            '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{input}<|END_OF_TURN_TOKEN|>'
            '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'),
        SUFFIX='<|END_OF_TURN_TOKEN|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|END_OF_TURN_TOKEN|>']),
    llama3_chat=dict(
        SYSTEM=('<|start_header_id|>system<|end_header_id|>\n\n'
                '{system}<|eot_id|>'),
        INSTRUCTION=(
            '<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>'
            '<|start_header_id|>assistant<|end_header_id|>\n\n'),
        SUFFIX='<|eot_id|>',
        SUFFIX_AS_EOS=True,
        STOP_WORDS=['<|eot_id|>']),
    phi3_chat=dict(
        SYSTEM='<|system|>\n{system}<|end|>\n',
        INSTRUCTION='<|user|>\n{input}<|end|>\n<|assistant|>\n',
        SUFFIX='<|end|>',
        SUFFIX_AS_EOS=True,
        SEP='\n',
        STOP_WORDS=['<|end|>']),
)