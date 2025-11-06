#!/bin/bash
rm -rf weights_transformer_italy_plain_ensemble/
python -m pdb train.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:1 --test_run
#python train.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:1
#python train.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:1 --test_run
