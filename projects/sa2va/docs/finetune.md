# Fine-tuning Sa2VA

This document provides a guide for fine-tuning Sa2VA on your custom datasets. We use a simple image referring segmentation task as an example.

## 1. Data Preparation

For fine-tuning, you need to prepare your dataset in a specific format. We provide an example `annotations.json` file, which is structured as follows:

```json
[
    {
        "image": "image_filename.jpg",
        "mask": [
            [[x1, y1, x2, y2, ...]], 
            ...
        ],
        "text": [
            "description for mask 1",
            "description for mask 2",
            ...
        ]
    },
    ...
]
```

- `image`: The filename of the image.
- `mask`: A list of segmentation masks. Each mask is a list of polygons.
- `text`: A list of text descriptions corresponding to each mask.

You can download the example dataset used in our fine-tuning example from [this Hugging Face repository](https://huggingface.co/datasets/bitersun/Sa2VA-finetune-example).

Place your data in the following structure:

```
data/
└── my_data/
    ├── annotations.json
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
```

For other types of data, you may need to customize the data loading logic. You can refer to `projects/sa2va/datasets/sa2va_data_finetune.py` for our example implementation.

## 2. Convert hf format checkpoint to pth format

Please run the following script to convert:

```bash
python tools/convert_to_pth.py hf_model_path --save-path PATH_TO_SAVE_FOLDER --arch-type internvl # or qwen
```

## 3. Configuration

The main configuration file for fine-tuning is `projects/sa2va/configs/sa2va_finetune.py`. You may need to adjust the following parameters based on your setup:

- `path`: Path to the pretrained model you want to fine-tune.
- `pretrained_pth`: Path to the pth model from step 2.
- `DATA_ROOT`: The root directory of your dataset (e.g., `./data/`).
- `batch_size`, `accumulative_counts`, `dataloader_num_workers`: Training parameters.
- `lr`, `max_epochs`: Learning rate and number of training epochs.

The dataset loading is handled by `Sa2VAFinetuneDataset` in `projects/sa2va/datasets/sa2va_data_finetune.py`. If your data format is different, you can create a custom dataset class and update the config file accordingly.

## 4. Start Fine-tuning

Once your data and configuration are ready, you can start the fine-tuning process using the following command:

```bash
bash tools/dist.sh train projects/sa2va/configs/sa2va_finetune.py 8
```

This command will start training on 8 GPUs. Adjust the number of GPUs as needed.
