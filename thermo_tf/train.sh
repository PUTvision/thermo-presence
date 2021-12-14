#!/bin/bash

CONFIG_PATH='./configs/config.yaml'

for FILTERS in 32 16
do
    echo "Filters:" $FILTERS
    python3 train.py --config_path $CONFIG_PATH --in_out_filters $FILTERS --log_neptune True
done

rm model_summary.txt output_frame.png *confusion_matrix.png
