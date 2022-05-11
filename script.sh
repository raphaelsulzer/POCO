#!/bash/bin

# python train.py --config configs/config_modelnet_viewpoint.yaml --log_mode interactive --experiment_name withSensor --normals True
# python train.py --config configs/config_modelnet_viewpoint.yaml --log_mode interactive --experiment_name noSensor --normals False

# python generate.py --config results/ModelNetViewpoints_withSensor_FKAConv_InterpAttentionKHeadsNet_None/config.yaml --gen_resolution_global 128
# python generate.py --config results/ModelNetViewpoints_noSensor_FKAConv_InterpAttentionKHeadsNet_None/config.yaml --gen_resolution_global 128

# python eval_meshes.py --gendir results/ModelNetViewpoints_withSensor_FKAConv_InterpAttentionKHeadsNet_None/gen_ModelNetViewpoints_test_3000/ --gtdir data/ModelNet10/ --dataset ModelNetViewpoints --meshdir meshes
# python eval_meshes.py --gendir results/ModelNetViewpoints_noSensor_FKAConv_InterpAttentionKHeadsNet_None/gen_ModelNetViewpoints_test_3000/ --gtdir data/ModelNet10/ --dataset ModelNetViewpoints --meshdir meshes