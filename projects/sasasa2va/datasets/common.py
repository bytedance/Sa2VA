SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask?",
    "What is {class_name} in this image? Please output segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",
]


SEG_STR = ", ".join([f"Clip_{i} [SEG]" for i in range(1, 10 + 1)]) + "."

ANSWER_LIST = [
    "It is " + SEG_STR,
    "Sure, " + SEG_STR,
    "Sure, it is " + SEG_STR,
    "Sure, the segmentation result is " + SEG_STR,
    SEG_STR,
]