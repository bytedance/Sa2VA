import numpy as np
import random
from xtuner.utils import DEFAULT_IMAGE_TOKEN

GCG_QUESTIONS = [
    DEFAULT_IMAGE_TOKEN + 'Could you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Can you provide a brief description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Please briefly describe the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    DEFAULT_IMAGE_TOKEN + 'Could you give a brief explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you give me an brief explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    DEFAULT_IMAGE_TOKEN + 'Could you provide me with a briefly analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

def refcocog_parse_annotations(example):
    # example {'id': str, 'refs': [{"setence", 'bbox', 'segmentation'},], 'img_file_name': str, 'caption': str}
    annotations = {'labels': [], 'caption': [], 'masks': [], 'tokens_positive': [],
                   'file_name': example['img_file_name'], 'image': example['img_file_name']}

    orig_caption = example['caption'].strip('"').strip()
    annotations['caption'] = orig_caption.lower()

    for detail in example['refs']:
        phrase = detail['sentence']
        if phrase.lower() in annotations['caption']:
            annotations['labels'].append(phrase)
            index = annotations['caption'].find(phrase)
            end_index = index + len(phrase) if index != -1 else -1
            annotations['tokens_positive'].append([index, end_index])
            # still polygon or rle
            annotations['masks'].append(detail["segmentation"])

    # Sort tokens_positive and corresponding lists
    tokens_positive = annotations['tokens_positive']
    sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
    annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
    annotations['masks'] = [annotations['masks'][i] for i in sorted_indices]
    annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]

    # Trimming overlapping intervals
    for i in range(len(tokens_positive)):
        for j in range(i + 1, len(tokens_positive)):
            # If there is overlap
            if tokens_positive[i][1] >= tokens_positive[j][0]:
                # Modify the end index of phrase i to be one less than the start index of phrase j
                tokens_positive[i][1] = tokens_positive[j][0] - 1
                # Modify the phrases to reflect the change in indices
                annotations['labels'][i] = orig_caption[tokens_positive[i][0]:tokens_positive[i][1] + 1]
                break  # Exit inner loop since i was modified

    return annotations

def refcocog_conversation(caption, tokens_positive):
    # insert <p> </p> and [seg] to caption and select a question
    question = random.choice(GCG_QUESTIONS).strip()

    # Prepare caption with tags
    def tag_caption(caption, tokens):
        for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
        return caption

    detailed_answer = tag_caption(caption, tokens_positive)

    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': detailed_answer}]
    return conversations

def refcocog_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    # Function to sort elements based on the start index of each phrase
    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    # Sort phrases based on their appearance in the sentence
    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = refcocog_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive

    return example

def glamm_refcocog_map_fn(example):
    # example {'id': str, 'refs': [{"setence", 'bbox', 'segmentation'},], 'img_file_name': str, 'caption': str}

    example = refcocog_parse_annotations(example)
    # example 'labels': [], 'caption': str, 'masks': [], 'tokens_positive': [], 'file_name': image_file

    example = refcocog_preprocess(example)

    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation += [{"from": "human", "value": input}, {"from": "gpt", "value": msg['value']}]
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example

def grandf_parse_annotations(example):
    image_path = example['file_name']
    annotations = {
        'labels': [], 'caption': [], 'masks': [],
        'tokens_positive': [], 'file_name': image_path,
        'image': image_path}
    annotations['caption'] = example['caption'].strip('"').strip()

    for word, grounding in example["groundings"].items():
        if grounding is None:
            continue
        annotations['labels'].append(word)
        annotations['tokens_positive'].append(grounding["token_positives"])
        annotations['masks'].append(grounding["rle_masks"])

    return annotations

def grandf_conversation(caption, tokens_positive):
    question = random.choice(GCG_QUESTIONS).strip()

    # Prepare caption with tags
    def tag_caption(caption, tokens):
        for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
        return caption

    detailed_answer = tag_caption(caption, tokens_positive)

    conversations = [{'from': 'human', 'value': question}, {'from': 'gpt', 'value': detailed_answer}]
    return conversations
def grandf_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    # Function to sort elements based on the start index of each phrase
    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    # Sort phrases based on their appearance in the sentence
    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = grandf_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive
    return example

def glamm_granf_map_fn(example):
    # example {'file_name': str, "height": int, "width": int, "image_id": str, caption: "str",
    # "groundings": {ground_words: {'token_positives', 'rle_masks', }}}
    example = grandf_parse_annotations(example)
    # example 'labels': [], 'caption': str, 'masks': [], 'tokens_positive': [], 'file_name': image_file

    example = grandf_preprocess(example)

    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation += [{"from": "human", "value": input}, {"from": "gpt", "value": msg['value']}]
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example

glamm_openpsg_map_fn = glamm_granf_map_fn

def flickr_parse_annotations(example):
    annotations = {'bboxes': [], 'labels': [], 'bboxes_ignore': [], 'caption': example['caption'], 'masks': [],
                   'tokens_positive': [], 'image': example['file_name']}
    ann_info = example["ann_info"]
    for ann in ann_info:
        if ann.get('ignore', False):
            continue
        x1, y1, w, h = ann['bbox']
        inter_w = max(0, min(x1 + w, example['width']) - max(x1, 0))
        inter_h = max(0, min(y1 + h, example['height']) - max(y1, 0))
        if inter_w * inter_h == 0 or ann['area'] <= 0 or w < 1 or h < 1:
            continue
        bbox = [x1, y1, x1 + w, y1 + h]
        annotations['bboxes'].append(bbox)
        tokens_positive = ann['tokens_positive']
        gt_label = [example['caption'][span[0]:span[1]] for span in tokens_positive]
        annotations['labels'].append(gt_label[0])
        annotations['tokens_positive'].append(tokens_positive[0])

        rle = ann['sam_mask']
        annotations['masks'].append(rle)

    # Convert bounding boxes to numpy arrays
    annotations['bboxes'] = np.array(annotations['bboxes'], dtype=np.float32) if annotations[
        'bboxes'] else np.zeros((0, 4), dtype=np.float32)
    annotations['bboxes_ignore'] = np.array(annotations['bboxes_ignore'], dtype=np.float32) if annotations[
        'bboxes_ignore'] else np.zeros((0, 4), dtype=np.float32)
    return annotations

def flickr_preprocess(example):
    data_labels = example['labels']
    masks = example['masks']
    caption = example['caption']
    tokens_positive = example['tokens_positive']

    # Function to sort elements based on the start index of each phrase
    def sort_by_start_index(items, order):
        return [items[i] for i in order]

    # Sort phrases based on their appearance in the sentence
    phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
    masks = sort_by_start_index(masks, phrase_order)
    data_labels = sort_by_start_index(data_labels, phrase_order)
    tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

    conversations = grandf_conversation(caption, tokens_positive)
    example['conversations'] = conversations
    example['labels'] = data_labels
    example['masks'] = masks
    example['tokens_positive'] = tokens_positive
    return example

def glamm_flickr_map_fn(example):
    # example {'file_name': str, "height": int, "width": int, "image_id": str, caption: "str",
    # "groundings": {ground_words: {'token_positives', 'rle_masks', }}}

    example = flickr_parse_annotations(example)

    example = flickr_preprocess(example)

    # do llava preprocess
    messages = example['conversations']
    input = ''
    conversation = []
    while messages and messages[0]['from'] == 'gpt':
        # Skip the first one if it is from gpt
        messages = messages[1:]
    for msg in messages:
        if msg['from'] == 'human':
            if DEFAULT_IMAGE_TOKEN in msg['value']:
                msg['value'] = msg['value'].replace(DEFAULT_IMAGE_TOKEN,
                                                    '').strip()
                msg['value'] = msg['value']
                msg['value'] = msg['value'].strip()
            input += msg['value']

        elif msg['from'] == 'gpt':
            conversation += [{"from": "human", "value": input}, {"from": "gpt", "value": msg['value']}]
            input = ''
        else:
            raise NotImplementedError
    example.update({'conversation': conversation})
    return example





