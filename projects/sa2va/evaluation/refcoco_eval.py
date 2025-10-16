import argparse
import copy
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np

from transformers import AutoModel, AutoTokenizer, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset

from projects.llava_sam2.models.utils import find_seg_indices

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
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
    parser.add_argument('--data_root', default='/mnt/bn/zilongdata-us/xiangtai/Sa2VA/data', help='Root directory for all datasets.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
    'grefcoco': {'splitBy': "unc", 'dataset_name': 'grefcoco'},
}

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle




def main():
    args = parse_args()

    image_folder = os.path.join(args.data_root, 'glamm_data/images/coco2014/train2014/')
    data_path = os.path.join(args.data_root, 'ref_seg/')

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

    if 'qwen' in args.model_path:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = None
    #model.preparing_for_generation(tokenizer, max_new_tokens=512)

    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info['dataset_name'],
        data_path=data_path,
        split=args.split,
        reasoning=False,
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
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        texts = data_batch['text']
        #print("texts:", texts)
        img_metas = {'img_id': data_batch['img_id'], 
                     'image_path': data_batch['image_path']}
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['image_path'], data_batch['text']
        pred_masks = []
        pred_texts = []
        pred_n_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred = model.predict_forward(**_data_batch, tokenizer=tokenizer, processor=processor)
            pred_mask = pred['prediction_masks']
            pred_text = pred['prediction']
            print(f"Text: {text}")
            # print(f"Raw pred text: '{repr(pred_text)}'")  # 使用 repr 查看完整格式
            # print(f"Pred text cleaned: '{pred_text.strip()}'")
            # print(f"Number of pred masks: {len(pred_mask)}")
            
            # 尝试清理文本
            cleaned_pred_text = pred_text.replace('<|im_end|>', '').strip()
            _, answer_seg_idx = find_seg_indices(cleaned_pred_text)
            #print(f"Answer seg indices with cleaned text: {answer_seg_idx}")

            if len(answer_seg_idx) == 0:
                # 如果清理后还是找不到，尝试原始文本
                _, answer_seg_idx = find_seg_indices(pred_text)
                #print(f"Answer seg indices with raw text: {answer_seg_idx}")
            
            pred_texts.append(pred_text)
            
            # whether the prediction is empty
            if len(pred_mask) == 0:
                #print('No mask predicted')
                pred_masks.append(None)
                pred_n_masks.append(None)
                continue
            else:
                # List (1, h, w) -> (n, h, w)
                pred_n_masks.append(np.concatenate(pred_mask, axis=0))
    
                # 清理预测文本
                cleaned_pred_text = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()
                #print(f"Cleaned pred_text: '{cleaned_pred_text}'")
                
                # 检查是否有[SEG]标记
                if '[SEG]' in cleaned_pred_text:
                    #print("Found [SEG] token, using predicted mask")
                    if len(pred_mask) > 0:
                        final_mask = pred_mask[0]
                        for mask in pred_mask[1:]:
                            final_mask = final_mask | mask
                        _ret_mask = mask_to_rle(final_mask)
                        pred_masks.append(_ret_mask)
                    else:
                        pred_masks.append(None)
                else:
                    #print("No [SEG] token found")
                    pred_masks.append(None)
                    continue

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

        if args.vis:
            import PIL
            import json
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

    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results)
        print(metric)

if __name__ == '__main__':
    main()
