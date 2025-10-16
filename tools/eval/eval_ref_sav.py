###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os.path as osp
import time
import argparse
import json
import numpy as np
import multiprocessing as mp
import pandas as pd
from pycocotools import mask as cocomask

from third_parts.revos.utils.metircs import db_eval_iou, db_eval_boundary, get_r2vos_accuracy, get_r2vos_robustness

NUM_WOEKERS = 128

def eval_queue(q, rank, out_dict):
    while not q.empty():
        vid_name, exp = q.get()
        vid = exp_dict[vid_name]
        exp_name = f'{vid_name}_{exp}'

        pred = mask_pred_dict[vid_name][exp]

        vid_len, h, w = len(vid['frames']), vid['height'], vid['width']
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            # all instances in the same frame
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)


            pred_mask = cocomask.decode(pred['prediction_masks'][frame_idx])
            pred_masks[frame_idx] += pred_mask
        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        a = get_r2vos_accuracy(gt_masks, pred_masks).mean()

        out_dict[exp_name] = [j, f, a]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str, )
    parser.add_argument("--exp_path", type=str, default="data/ref_sav_eval/meta_expressions_valid.json")
    parser.add_argument("--mask_path", type=str, default="data/ref_sav_eval/mask_dict.json")
    parser.add_argument("--save_json_name", type=str, default="ref_sav_valid.json")
    parser.add_argument("--save_csv_name", type=str, default="ref_sav_valid.csv")
    args = parser.parse_args()
    queue                = mp.Queue()
    exp_dict             = json.load(open(args.exp_path))['videos']
    mask_dict            = json.load(open(args.mask_path))

    mask_pred = json.load(open(args.pred_path))

    shared_exp_dict             = mp.Manager().dict(exp_dict)
    shared_mask_dict            = mp.Manager().dict(mask_dict)
    output_dict                 = mp.Manager().dict()

    mask_pred_dict  = mp.Manager().dict(mask_pred)


    for idx, vid_name in enumerate(exp_dict):
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict)

    # save average results
    output_json_path = osp.join(osp.dirname(args.pred_path), args.save_json_name)
    output_csv_path  = osp.join(osp.dirname(args.pred_path), args.save_csv_name)

    data_list = []
    for videxp, (j, f, a) in output_dict.items():
        vid_name, exp = videxp.rsplit('_', 1)
        data = {}

        data['video_name'] = vid_name
        data['exp_id']     = exp
        data['exp']        = exp_dict[vid_name]['expressions'][exp]['exp']
        data['videxp']     = videxp
        data['J']          = round(100 * j, 2)
        data['F']          = round(100 * f, 2)
        data['JF']         = round(100 * (j + f) / 2, 2)
        data['A']          = round(100 * a, 2)
        data['type_id']    = exp_dict[vid_name]['expressions'][exp]['type_id']

        data_list.append(data)

    is_long = lambda x: x['type_id'] == 0
    is_short    = lambda x: x['type_id'] == 1

    j_referring  = np.array([d['J'] for d in data_list if is_long(d)]).mean()
    f_referring  = np.array([d['F'] for d in data_list if is_long(d)]).mean()
    a_referring  = np.array([d['A'] for d in data_list if is_long(d)]).mean()
    jf_referring = (j_referring + f_referring) / 2

    j_reason  = np.array([d['J'] for d in data_list if is_short(d)]).mean()
    f_reason  = np.array([d['F'] for d in data_list if is_short(d)]).mean()
    a_reason  = np.array([d['A'] for d in data_list if is_short(d)]).mean()
    jf_reason = (j_reason + f_reason) / 2

    j_referring_reason  = (j_referring + j_reason) / 2
    f_referring_reason  = (f_referring + f_reason) / 2
    a_referring_reason  = (a_referring + a_reason) / 2
    jf_referring_reason = (jf_referring + jf_reason) / 2

    results = {
        "long": {
            "J" : j_referring,
            "F" : f_referring,
            "A" : a_referring,
            "JF": jf_referring
        },
        "short": {
            "J" : j_reason,
            "F" : f_reason,
            "A" : a_reason,
            "JF": jf_reason
        },
        "overall": {
            "J" : j_referring_reason,
            "F" : f_referring_reason,
            "A" : a_referring_reason,
            "JF": jf_referring_reason
        }
    }

    print(results)
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_json_path}")

    data4csv = {}
    for data in data_list:
        for k, v in data.items():
            data4csv[k] = data4csv.get(k, []) + [v]

    df = pd.DataFrame(data4csv)
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))