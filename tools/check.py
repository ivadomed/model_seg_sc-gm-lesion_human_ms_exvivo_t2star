import pandas as pd
import pathlib

base = pathlib.Path("/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results/Dataset1718_MagPhase_patchsize_5_adamw/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres/inference_results/patchsize_5_adamw_validation_EVAL")

single_dirs = [d for d in base.glob("fold_*") if d.is_dir() and "TTA" not in d.name and "Ensemble" not in d.name]
tta_dirs = [d for d in base.glob("fold_*_TTA") if d.is_dir()]

print("SINGLE")
m1 = []
for d in sorted(single_dirs):
    df = pd.read_csv(d / "metrics_2d_global.csv")
    val = df[df["Label"] == 1]["Global_Dice"].values[0]
    print(val)
    m1.append(val)
print("Mean:", sum(m1)/len(m1))

print("TTA")
m2 = []
for d in sorted(tta_dirs):
    df = pd.read_csv(d / "metrics_2d_global.csv")
    val = df[df["Label"] == 1]["Global_Dice"].values[0]
    print(val)
    m2.append(val)
print("Mean:", sum(m2)/len(m2))
