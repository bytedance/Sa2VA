from eve.eval.run_eve import eval_model

model_path = "Absolute Path of BAAI/EVE-7B-HD-v2.0"
model_type = 'qwen2'
conv_mode = 'qwen2'

prompt = "Please describe the image in detail."
image_file = "examples/ocr_beijing.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_type": model_type,
    "query": prompt,
    "conv_mode": conv_mode,
    "image_file": image_file,
    "temperature": 0.2,
    "do_sample": True,
    "top_p": None,
    "top_k": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)