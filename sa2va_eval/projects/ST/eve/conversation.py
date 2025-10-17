import dataclasses
from enum import Enum, auto
from typing import List


class SeparatorStyle(Enum):
    """Different separator style."""
    TWO = auto()
    PLAIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"

    skip_next: bool = False

    def get_prompt(self):
        messages = self.messages
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].replace("<image>", "").strip()
            messages[0] = (init_role, "<image>\n" + init_msg)

        if self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_images(self, return_pil=False):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO

                    from PIL import Image
                    msg, image, image_process_mode = msg
                    if image_process_mode == "Pad":
                        def expand2square(pil_img, background_color=(122, 116, 104)):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color)
                                result.paste(
                                    pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color)
                                result.paste(
                                    pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image)
                    elif image_process_mode in ["Default", "Crop"]:
                        pass
                    elif image_process_mode == "Resize":
                        image = image.resize((336, 336))
                    else:
                        raise ValueError(
                            f"Invalid image_process_mode: {image_process_mode}")
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(
                        min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if longest_edge != max(image.size):
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                    if return_pil:
                        images.append(image)
                    else:
                        buffered = BytesIO()
                        image.save(buffered, format="PNG")
                        img_b64_str = base64.b64encode(
                            buffered.getvalue()).decode()
                        images.append(img_b64_str)
        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple:
                    import base64
                    from io import BytesIO
                    msg, image, image_process_mode = msg
                    max_hw, min_hw = max(image.size), min(image.size)
                    aspect_ratio = max_hw / min_hw
                    max_len, min_len = 800, 400
                    shortest_edge = int(
                        min(max_len / aspect_ratio, min_len, min_hw))
                    longest_edge = int(shortest_edge * aspect_ratio)
                    W, H = image.size
                    if H > W:
                        H, W = longest_edge, shortest_edge
                    else:
                        H, W = shortest_edge, longest_edge
                    image = image.resize((W, H))
                    buffered = BytesIO()
                    image.save(buffered, format="JPEG")
                    img_b64_str = base64.b64encode(
                        buffered.getvalue()).decode()
                    img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                    msg = img_str + msg.replace('<image>', '').strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        if len(self.get_images()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }

conv_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llama_v3 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="llama3",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|end_of_text|>",
)

conv_qwen_v2 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="qwen2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="<|im_end|>",
)

default_conversation = conv_qwen_v2
conv_templates = {
    "default": conv_qwen_v2,
    "plain": conv_plain,
    "llama3": conv_llama_v3,
    "qwen2": conv_qwen_v2,
}


if __name__ == "__main__":
    # print(default_conversation.get_prompt())
    import torch, transformers
    from eve.train.train import preprocess_plain, preprocess_eve
    from eve.model import *

    # version = 'llama3'
    # sample_type = 'text'
    # model_type = 'llama3'
    # model_path = 'lmsys/Llama-3-8B-Instruct'

    version = 'qwen2'
    sample_type = 'image'
    model_type = 'qwen2'
    model_path = 'lmsys/Qwen2-7B-Instruct'

    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
    )

    if model_type == 'llama3':
        tokenizer.pad_token = "<|reserved_special_token_0|>"
        tokenizer.pad_token_id = 128002

    print(tokenizer.bos_token, tokenizer.bos_token_id)
    print(tokenizer.eos_token, tokenizer.eos_token_id)
    print(tokenizer.pad_token, tokenizer.pad_token_id)

    assert tokenizer.eos_token_id and tokenizer.pad_token_id
    assert tokenizer.eos_token_id != tokenizer.pad_token_id

    if version == 'plain':
        if sample_type == 'image':
            sources = [[{'from': 'human', 'value': 'help me <image>\nWhat?'}, 
            {'from': 'gpt', 'value': 'The bus is coming.'}
            ]]
        else:
            sources = [[
                {"from": "human", "value": "What is the main activity happening in the image?"},
                {"from": "gpt", "value": "The main activity happening in the image."},
            ]]

        print('has image:', '<image>' in sources[0][0]["value"], '\n')
        output = preprocess_plain(sources, tokenizer, has_image=('<image>' in sources[0][0]["value"]))
    else:
        if sample_type == 'image':
            sources = [[{'from': 'human', 'value': '<image>\nWhat feature?'}, 
            {'from': 'gpt', 'value': 'The bus is coming.'},
            {'from': 'human', 'value': 'What feature?'}, 
            {'from': 'gpt', 'value': 'The back bus is coming.'},
            ]]
        else:
            sources = [[
                {"from": "human", "value": "What is the main activity?"},
                {"from": "gpt", "value": "The main activity in the image."},
                {"from": "human", "value": "What is the main?"},
                {"from": "gpt", "value": "The main activity."}
            ]]

        print('has image:', '<image>' in sources[0][0]["value"], '\n')
        output = preprocess_eve(sources, tokenizer, has_image=('<image>' in sources[0][0]["value"]))

    print('input_ids', output['input_ids'])
    print('labels', output['labels'], '\n')

    input_ids = [tokenizer.pad_token_id if x == -200 else x for x in output['input_ids'][0].tolist()]
    print(tokenizer.convert_ids_to_tokens(input_ids))
    labels = [tokenizer.pad_token_id if x == -100 else x for x in output['labels'][0].tolist()]
    print(tokenizer.convert_ids_to_tokens(labels))
    assert len(input_ids) == len(labels)
    