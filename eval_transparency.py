import os
from argparse import ArgumentParser

ROOT_DIR = "/home/constantin/data/anatomy_splatting"
ADD_STANDALONE = True

# Scene layer mapping
scenes = ["TransparencyTest/L0", "TransparencyTest/L0-half"]

out = "eval_gen1/TransparencyTest"

transparency_modes = {"N": " ", "R": " --random_background ", "A": " -a ", "RA": " --random_background -a "}
common_args = " --eval --test_iterations -1 " #--quiet "

eval_models = []

for scene in scenes:
    for mode, cmd in transparency_modes.items():
        source = os.path.join(ROOT_DIR, "renders", scene)
        outpath = os.path.join(ROOT_DIR, out, f"{scene}-{mode}")
        os.system(f"python train.py -s {source} -m {outpath} {common_args} {cmd}")
        eval_models.append(outpath)

for model in eval_models:
    os.system(f"python render.py --iteration 7000 -m {model}")
    os.system(f"python render.py --iteration 30000 -m {model}")

for model in eval_models:
    os.system(f"python metrics.py -m {model}")