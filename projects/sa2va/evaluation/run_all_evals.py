import os
import argparse
import subprocess

def run_command(command):
    """Executes a command in the shell and prints its output."""
    print(f"Executing: {' '.join(command)}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.stdout.close()
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)

def main():
    parser = argparse.ArgumentParser(description="Run all evaluation scripts.")
    parser.add_argument('model_path', help='HF model path.')
    parser.add_argument('--data_root', default='./data', help='Root directory for all datasets.')
    parser.add_argument('--gpus', type=str, default='8', help='Number of GPUs to use for distributed testing.')
    args = parser.parse_args()

    base_path = "./projects/sa2va/evaluation"
    dist_test_script = "./projects/sa2va/evaluation/dist_test.sh"

    # --- Evaluation Commands ---
    eval_configs = {
        "RefCOCO": {
            "script": os.path.join(base_path, "sa2va_eval_refcoco.py"),
            "datasets": ["refcoco", "refcoco_plus", "refcocog"],
            "split": "test"
        },
        "GCG": {
            "script": os.path.join(base_path, "sa2va_eval_gcg.py"),
            "split": "val",
            "metrics_script": os.path.join(base_path, "metrics_gcg.py"),
        },
        "RefVOS": {
            "script": os.path.join(base_path, "sa2va_eval_ref_vos.py"),
            "datasets": ["DAVIS", "MEVIS_U"],
        }
    }

    try:
        # --- RefCOCO ---
        refcoco_config = eval_configs["RefCOCO"]
        for dataset in refcoco_config["datasets"]:
            print(f"\n----- Running RefCOCO evaluation for {dataset} -----")
            script_args = f"--dataset={dataset} --split={refcoco_config['split']} --data_root={args.data_root}"
            cmd = ["bash", dist_test_script, refcoco_config["script"], args.model_path, args.gpus] + script_args.split()
            run_command(cmd)

        # --- GCG ---
        gcg_config = eval_configs["GCG"]
        print("\n----- Running GCG evaluation -----")
        gcg_pred_dir = f"./gcg_pred/{os.path.basename(args.model_path)}"
        script_args_gcg = f"--split={gcg_config['split']} --save_dir={gcg_pred_dir} --data_root={args.data_root}"
        cmd_gcg = ["bash", dist_test_script, gcg_config["script"], args.model_path, args.gpus] + script_args_gcg.split()
        run_command(cmd_gcg)
        print("\n----- Calculating GCG metrics -----")
        cmd_gcg_metrics = ["python", gcg_config["metrics_script"], f"--split={gcg_config['split']}", f"--prediction_dir_path={gcg_pred_dir}", f"--gt_dir_path={os.path.join(args.data_root, 'glamm_data/annotations/gcg_val_test/')}"]
        run_command(cmd_gcg_metrics)

        # --- RefVOS ---
        refvos_config = eval_configs["RefVOS"]
        for dataset in refvos_config["datasets"]:
            print(f"\n----- Running RefVOS evaluation for {dataset} -----")
            work_dir = f"work_dirs/{os.path.basename(args.model_path)}"
            script_args_vos = f"--dataset={dataset} --work_dir={work_dir} --data_root={args.data_root}"
            cmd_vos = ["bash", dist_test_script, refvos_config["script"], args.model_path, args.gpus] + script_args_vos.split()
            run_command(cmd_vos)

        print("\nAll evaluations completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"\nAn error occurred during evaluation: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()