import os

ROOT_DIR = "/home/constantin/data/anatomy_splatting"
ADD_STANDALONE = False

# Scene layer mapping
ct_scenes = {
    "leg":["bone", "muscle", "full"],
    "fullbody": ["bone", "organs", "muscle"],
    "lowerbody": ["L0-bone", "L1-muscle", "L2-tissue"]}

synthetic_scenes = {
    "2Layer": ["L0", "L1"],
    "3Layer": ["L0", "L1", "L2"],
    "TransparencyTest": ["L0"]}

all_scenes = {**ct_scenes, **synthetic_scenes}

model_out = "model_gen"
eval_out = "eval_gen"

quality_levels = {"default": " ", "HQ": " --densify_grad_threshold 0.0001 --percent_dense 0.005"}
#quality_levels = {"low": " --densify_grad_threshold 0.0002 --percent_dense 0.01 ", "lower": " --densify_grad_threshold 0.001 --percent_dense 0.03 "}

common_args = "--random_background -a --save_layer --test_iterations -1 " #--quiet "

eval_models = []

for mode in [model_out, eval_out]:
    for q_level, q_cmd in quality_levels.items():
        for scene, layers in all_scenes.items():
            for idx, layer in enumerate(layers):
                source = os.path.join(ROOT_DIR, "renders", scene, layer)
                outpath = os.path.join(ROOT_DIR, mode, scene, f"{layer}-{q_level}")
                additional_args = q_cmd

                if "eval" in mode:
                    eval_models.append(outpath)
                    additional_args += " --eval "

                if idx > 0:
                    if ADD_STANDALONE:
                        os.system(f"python train.py -s {source} -m {outpath}-standalone {common_args} {additional_args}")
                        if "eval" in mode:
                            eval_models.append(f"{outpath}-standalone")
 
                    additional_args += " --load_layer " + os.path.join(ROOT_DIR, mode, scene, f'{layers[idx - 1]}-{q_level}', "layer_save_30000.pth")
                
                os.system(f"python train.py -s {source} -m {outpath} {common_args} {additional_args}")

for model in eval_models:
    os.system(f"python render.py --iteration 7000 -m {model}")
    os.system(f"python render.py --iteration 30000 -m {model}")

for model in eval_models:
    os.system(f"python metrics.py -m {model}")