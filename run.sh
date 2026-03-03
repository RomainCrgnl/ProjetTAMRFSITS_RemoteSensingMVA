TORCH_NUM_THREADS=4 PYTHONOPTIMIZE=TRUE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tamrfsits/bin/test.py \
--ts ${PWD}/dataset/test/ \
--output ${PWD}/gap_filling/tamrfsits \
--checkpoint ${PWD}/model/tamrfsits_pretrained_2015974.ckpt \
--config ${PWD}/model/hydra_config/ \
--algorithm TAMRFSITS \
--strategy FORECAST \
--width 1650 \
--subtile_width 165 \
--forecast_doy_start 355 \
--write_images \
--device cpu \
--disable_metrics \
--show_subtile_progress \
#--generate_animation
