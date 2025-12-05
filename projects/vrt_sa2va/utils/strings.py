import re


# directly copy from llava_sam2/models/utils.py
def find_seg_indices(text):
    all_seg_indices = [m.start() for m in re.finditer(r'\[SEG\]', text)]
    answer_spans = [(m.start(), m.end()) for m in re.finditer(r'<answer>.*?</answer>', text, re.DOTALL)]
    if len(answer_spans) == 0:
        return [], []
    if len(answer_spans) > 1:
        print(f"Warning: There should be only one <answer> tag in the text. {text}")
        # raise ValueError(f"There should be only one <answer> tag in the text. {text}")
    answer_span = answer_spans[0]
    start, end = answer_span
    
    seg_indices_in_reason = []
    seg_indices_in_answer = []
    for idx, seg_ind in enumerate(all_seg_indices):
        if start <= seg_ind < end:
            seg_indices_in_answer.append(idx)
        elif seg_ind < start:
            seg_indices_in_reason.append(idx)
        else:
            continue
    return seg_indices_in_reason, seg_indices_in_answer