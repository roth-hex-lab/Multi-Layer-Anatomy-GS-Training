import os
from argparse import ArgumentParser

ROOT_DIR = "/home/constantin/data/anatomy_splatting"

# Evaluate purging parameters
purging_scenes_muscle = ["leg/muscle"]
purging_scenes_full = ["leg/full-v1", "leg/full-v2"]

purging_thresholds = [0.005, 0.01, 0.015, 0.02, 0.03]
purging_decays = [0.9, 0.94, 0.96, 0.975, 0.99, 0.999]

purging_out_dir = "eval/purge"
common_args = " --random_background -a --eval --test_iterations 100 350 750 1500 3500 7000 16000 30000"

purge_models = []

for threshold in purging_thresholds:
    for decay in purging_decays:

        for muscle in purging_scenes_muscle:
            source = os.path.join(ROOT_DIR, "renders", muscle)
            outpath = os.path.join(ROOT_DIR, "eval/purge", f"{muscle}_d_{decay}_t_{threshold}")
            load_layer = os.path.join(ROOT_DIR, "models/leg/bone/layer_save_30000.pth")
            os.system(f"python train.py -s {source} -m {outpath} --load_layer {load_layer} --inactive_purge_threshold {threshold} --inactive_purge_decay {decay} {common_args}")
            purge_models.append(outpath)

        for full in purging_scenes_full:
            source = os.path.join(ROOT_DIR, "renders", full)
            outpath = os.path.join(ROOT_DIR, "eval/purge", f"{full}_d_{decay}_t_{threshold}")
            load_layer = os.path.join(ROOT_DIR, "models/leg/muscle/layer_save_30000.pth")
            os.system(f"python train.py -s {source} -m {outpath} --load_layer {load_layer} --inactive_purge_threshold {threshold} --inactive_purge_decay {decay} {common_args}")
            purge_models.append(outpath)

for scene in purge_models:
    source = os.path.join(ROOT_DIR, "eval/purge", scene)
    os.system(f"python render.py --iteration 7000 -m {source}")
    os.system(f"python render.py --iteration 30000 -m {source}")

for scene in purge_models:
    source = os.path.join(ROOT_DIR, "eval/purge", scene)
    os.system(f"python metrics.py -m {source}")