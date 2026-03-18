BASE_DIR=forecasting
mkdir -p $BASE_DIR

# output folder selection
LAST_RUN=$(ls -d $BASE_DIR/run_* 2>/dev/null | sed 's/.*run_//' | sort -n | tail -n 1)

if [ -z "$LAST_RUN" ]; then
  RUN_ID=1
else
  RUN_ID=$((10#$LAST_RUN + 1))
fi

RUN_DIR=$(printf "%s/run_%03d" "$BASE_DIR" "$RUN_ID")
mkdir -p "$RUN_DIR"

echo "Output folder: $RUN_DIR"

# command
TORCH_NUM_THREADS=4 PYTHONOPTIMIZE=TRUE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tamrfsits/bin/test.py \
--ts ${PWD}/dataset/31UES_12/ \
--output ${PWD}/$RUN_DIR \
--checkpoint ${PWD}/model/tamrfsits_pretrained_2015974.ckpt \
--config ${PWD}/model/hydra_config/ \
--algorithm TAMRFSITS \
--strategy CUSTOM_FORECAST \
--width 1650 \
--patch_idx 1 \
--subtile_width 165 \
--margin 1 \
--forecast_doy_start 318 \
--custom_forecast_context_size 5 \
--custom_forecast_gap_step 1 \
--custom_forecast_only_hr \
--dt_orig 2022-01-01 \
--show_subtile_progress \
--device cpu \
--write_images \
--disable_metrics \
--generate_animation \
--custom_forecast_sliding_window \

# --generate_animation \
#--custom_forecast_only_hr \
#--patch_idx # index du patch dans l'image, surement [0, 35]
#--disable_metrics \


# INFOS:
# 30SWH_24: --patch_idx: 19 / 25 ; --margin 1 (images 11 à 12)
# 31TCJ_12: --patch_idx: 19 ; --margin 1 (images 8 à 22)
# 31UES_12: --patch_idx 1 ; --margin 1 (images 5 à 8)