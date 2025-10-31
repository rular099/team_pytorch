#!/bin/bash
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnode=1 --nproc-per-node=2 main_custom.py --conf_file ./config/conf_reg.yml
