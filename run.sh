BASE_DIR=forecasting
mkdir -p $BASE_DIR

# output folder selection
LAST_RUN=$(ls -d $BASE_DIR/run_* 2>/dev/null | sed 's/.*run_//' | sort -n | tail -n 1)

if [ -z "$LAST_RUN" ]; then
  RUN_ID=1
else
  RUN_ID=$((LAST_RUN + 1))
fi

RUN_DIR=$(printf "%s/run_%03d" "$BASE_DIR" "$RUN_ID")
mkdir -p "$RUN_DIR"

echo "Output folder: $RUN_DIR"

# command
TORCH_NUM_THREADS=4 PYTHONOPTIMIZE=TRUE PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python tamrfsits/bin/test.py \
--ts ${PWD}/dataset/test/ \
--output ${PWD}/$RUN_DIR \
--checkpoint ${PWD}/model/tamrfsits_pretrained_2015974.ckpt \
--config ${PWD}/model/hydra_config/ \
--algorithm TAMRFSITS \
--strategy FORECAST \
--width 1650 \
--subtile_width 165 \
--margin 30 \
--forecast_doy_start 327 \
--show_subtile_progress \
--device cpu \
--write_images \
--generate_animation
#--dt_orig 2022-01-01 \
#--disable_metrics \
