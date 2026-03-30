#!/bin/bash
rm -rf weights_transformer_italy_plain_ensemble/
TEST_RUN_FLAG=""
if [ "${TEST_RUN:-0}" = "1" ]; then
  TEST_RUN_FLAG="--test_run"
fi
OVERFIT_FLAG=""
if [ "${OVERFIT_N:-0}" != "0" ]; then
  OVERFIT_FLAG="--overfit_n ${OVERFIT_N}"
fi

if [ "${OVERFIT_N:-0}" != "0" ]; then
  NPROC_PER_NODE=${NPROC_PER_NODE:-1}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
else
  NPROC_PER_NODE=${NPROC_PER_NODE:-4}
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  train_light.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml ${TEST_RUN_FLAG} ${OVERFIT_FLAG}
# debug: python -m pdb train_light.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:3 --test_run
