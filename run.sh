export CUDA_VISIBLE_DEVICES=1
export scene=lego
python train.py --group=nerfacc --model=tensorf --yaml=tensorf_blender --name=$scene --data.scene=$scene --barf_c2f=[0.1,0.5] --visdom!