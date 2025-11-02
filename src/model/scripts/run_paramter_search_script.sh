#!/bin/bash

# alredy done 2023. 03.29
# check batch size 
if false; then
lr=0.0005
python build_basemodel.py --model lstm --lr $lr --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size 128 --case lr0005_embed128_layer2_hdim256_batch128
python build_basemodel.py --model lstm --lr 0.001 --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size 128 --case lr001_embed128_layer2_hdim256_batch128
python build_basemodel.py --model lstm --lr $lr --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size 256 --case lr0005_embed128_layer2_hdim256_batch256
python build_basemodel.py --model lstm --lr $lr --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size 512 --case lr0005_embed128_layer2_hdim256_batch512
python build_basemodel.py --model lstm --lr $lr --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size 1024 --case lr0005_embed128_layer2_hdim256_batch1024
fi

if false; then
bs=128
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 64 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed64_layer2_hdim256
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 2 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size $bs  --case lr0005_embed128_layer2_hdim256 
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 3 --hidden-dim 256 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer3_hdim256
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 3 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer3_hdim512
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 3 --hidden-dim 512 --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer3_hdim512_nonlayernorm

# added one more
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 256 --nlayers 3 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed256_layer3_hdim512
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 256 --nlayers 4 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed256_layer4_hdim512
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 64  --nlayers 4 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed64_layer4_hdim512
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 4 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer4_hdim512

python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 512 --nlayers 4 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed512_layer4_hdim512
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 128 --nlayers 5 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer5_hdim512
fi

bs=128
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 256 --nlayers 5 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed128_layer5_hdim256
python build_basemodel.py --model lstm --lr 0.0005 --embed-dim 512 --nlayers 5 --hidden-dim 512 --layernorm --dropout-ratio 0.2 --batch-size $bs --case lr0005_embed512_layer5_hdim512
