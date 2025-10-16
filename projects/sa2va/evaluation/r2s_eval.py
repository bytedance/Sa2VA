import argparse
import copy
import json
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np

from transformers import AutoModel, AutoTokenizer

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import R2SDataset

from projects.llava_sam2.models.utils import find_seg_indices


DATASETS_ATTRIBUTES = {
    "VER": {
        'json_path': 'data/dense/eval_ann_138.json', 
        'dataset_name': 'VER',
    },
    
}

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='VER',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--vis',
        action='store_true',
        help='whether to visualize the results')
    
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--data_root', default='./data', help='Root directory for all datasets.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


IMAGE_FOLDER = './data/dense/images'
DATA_PATH = './data/dense/'

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def main():
    args = parse_args()

    image_folder = os.path.join(args.data_root, 'dense/images')
    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset_info['json_path'] = os.path.join(args.data_root, 'dense/eval_ann_138.json')

    if args.launcher != 'none':
        import datetime
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model.preparing_for_generation(tokenizer, max_new_tokens=512)

    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset = R2SDataset(
        image_folder=image_folder,
        dataset_name=dataset_info['dataset_name'],
        json_path=dataset_info['json_path'],
        split=args.split,
        reasoning=True,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )

    results = []
    cnt = 0
    model_name = args.model_path.strip('/').split('/')[-1]
    for data_batch in tqdm.tqdm(dataloader):
        prediction = {
            'img_id': data_batch['img_id'],
            'image_path': data_batch['image_path'],
            'reasoning': data_batch['reasoning'],
            'answers': data_batch['answers'],
        }
        texts = [data_batch['text']]
        img_metas = {'img_id': data_batch['img_id'], 
                     'image_path': data_batch['image_path']}
    
        del data_batch['img_id'], data_batch['image_path'], data_batch['reasoning'], data_batch['answers'], data_batch['text']
    
        pred_masks = []
        pred_texts = []
        pred_n_masks = []
        reasoning_masks = []
    
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred = model.predict_forward(**_data_batch, tokenizer=tokenizer)
            pred_mask = pred['prediction_masks']
            pred_text = pred['prediction']
            #print(f"Pred text: {pred_text}")
            pred_texts.append(pred_text)
            
            if len(pred_mask) == 0:
                #print('No mask predicted')
                pred_masks.append(None)
                pred_n_masks.append(None)
                reasoning_masks.append(None)
                continue
            else:
                pred_n_masks.append(np.concatenate(pred_mask, axis=0))
                
                cleaned_pred_text = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()

                if '[SEG]' in cleaned_pred_text:
                    #print("Found [SEG] token, using predicted mask")
                    if len(pred_mask) > 0:
                        pred_masks.append(pred_mask)
                        reasoning_masks.append(None)
                    else:
                        pred_masks.append(None)
                        reasoning_masks.append(None)
                else:
                    #print("No [SEG] token found")
                    pred_masks.append(None)
                    reasoning_masks.append(None)
    
        prediction.update({
            'prediction_masks': pred_masks,
            'prediction_texts': pred_texts,
            'reasoning_masks': reasoning_masks
        })
        results.append(prediction)


        if args.vis:
            import PIL
            from projects.llava_sam2.evaluation.utils.visualization import visualize_n_masks, visualize_mask
            img_id = img_metas['img_id']
            image_path = img_metas['image_path']
            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]
            img = PIL.Image.open(image_path).convert('RGB')
            for i, pred_n_mask in enumerate(pred_n_masks):
                if pred_n_mask is None:
                    continue
                
                mask_name = f'{image_name}_{i}.png'
                mask_path = os.path.join('./work_dirs/vis', model_name, args.dataset, mask_name)
                if not os.path.exists(mask_path):
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)

                # visualize
                pred_n_mask = [
                    PIL.Image.fromarray(pred_1_mask.astype(np.float32) * 255).convert('L')
                    for pred_1_mask in pred_n_mask
                ]
                # pred_mask_vis = visualize_n_masks(img, pred_n_mask)
                # pred_mask_vis.save(mask_path)
                pred_mask_vis = [visualize_mask(img, pred_mask) for pred_mask in pred_n_mask]
                for idx, pred_mask_vis_cur in enumerate(pred_mask_vis):
                    pred_mask_vis_cur.save(mask_path.replace('.png', '_{:06d}.png'.format(idx)))

                text_output = pred_texts[i]
                text_input = texts[i]
                text_output = text_output.replace('<|im_end|>', '')
                text_json = {
                    'question': text_input,
                    'answer': text_output,
                }
                json_name = f'{image_name}_{i}.json'
                json_path = os.path.join('./work_dirs/vis', model_name, args.dataset, json_name)
                json.dump(
                    text_json,
                    open(json_path, 'w'),
                    indent=4,
                )

    
    tmpdir = 'work_dirs/' + args.dataset + '_' + args.model_path.replace('work_dirs', '').replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)


    if get_rank() == 0: 
        if results is not None:
            for item in results:
                if 'prediction_masks' in item and item['prediction_masks'] is not None:
                    new_pred_masks = []
                    for mask_group in item['prediction_masks']:
                        if mask_group is not None:
                            new_mask_group = [mask.tolist() for mask in mask_group]
                            new_pred_masks.append(new_mask_group)
                        else:
                            new_pred_masks.append(None)
                    item['prediction_masks'] = new_pred_masks
                if 'reasoning_masks' in item and item['reasoning_masks'] is not None:
                    new_reasoning_masks = []
                    for mask_group in item['reasoning_masks']:
                        if mask_group is not None:
                            new_mask_group = [mask.tolist() for mask in mask_group]
                            new_reasoning_masks.append(new_mask_group)
                        else:
                            new_reasoning_masks.append(None)
                    item['reasoning_masks'] = new_reasoning_masks
            #print("Rank 0: Conversion complete.")

            os.makedirs(tmpdir, exist_ok=True)
            #print(f"Rank 0: Saving results to {os.path.join(tmpdir, 'results.json')}")
            json.dump(
                results,
                open(os.path.join(tmpdir, 'results.json'), 'w'),
                indent=4,
            )
            print("Rank 0: Results saved successfully.")
        else:
            print("Rank 0: Received None as results, skipping file write.")

if __name__ == '__main__':
    main()
