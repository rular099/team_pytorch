#!/bin/bash
rm -rf weights_transformer_italy_plain_ensemble/
python -m pdb train_light.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:3 --test_run
#python train_light.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:1
#python train_light.py --config ./pga_configs/transformer_italy_plain_ensemble.json --diting_config ./diting/config/conf_reg.yml --device cuda:1 --test_run
