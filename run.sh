export CUDA_VISIBLE_DEVICES=1
export scene=lego

scene_list=(chair drums ficus hotdog lego materials mic ship)

for scene in ${scene_list[@]} 
do
    python train.py --group=nerfacc --model=tensorf --yaml=tensorf_blender --name=$scene --data.scene=$scene  --visdom!
done