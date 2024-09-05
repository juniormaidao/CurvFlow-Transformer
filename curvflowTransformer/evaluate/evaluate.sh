#    --dataset-source pyg \       --dataset-name zinc \
#     --user-data-dir ../customized_dataset \     --arch graphormer_base \
#     --dataset-name customized_qm9_dataset \  graphormer_slim
python evaluate.py \
    --user-dir ../../graphormer \
    --num-workers 0 \
    --ddp-backend=legacy_ddp \
    --user-data-dir ../customized_dataset \
    --dataset-name customized_qm9_dataset \
    --task graph_prediction \
    --criterion l1_loss \
    --arch graphormer_base \
    --num-classes 1 \
    --batch-size 12 \
    --pretrained-model-name pcqm4mv2_graphormer_base \
    --load-pretrained-model-output-layer \
    --max-nodes 128 \
    --split test \


#python evaluate.py \
#    --user-dir ../../graphormer \
#    --num-workers 0 \
#    --ddp-backend=legacy_ddp \
#    --dataset-name zinc \
#    --dataset-source pyg \
#    --task graph_prediction \
#    --criterion l1_loss \
#    --arch graphormer_slim \
#    --num-classes 1 \
#    --batch-size 32 \
#    --pretrained-model-name pcqm4mv2_graphormer_base \
#    --load-pretrained-model-output-layer \
#    --max-nodes 256 \
#    --split test \
#    --seed 1


