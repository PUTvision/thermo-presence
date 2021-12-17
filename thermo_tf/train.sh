#!/bin/bash

CONFIG_PATH='./configs/config.yaml'

for FILTERS in 64 32 16 8
do
    for BATCH_NORM in 1 0
    do
        for CONV_TRANSPOSE in 1 0
        do
            for SQUEEZE in 1 0
            do
                if [ "$SQUEEZE" -eq "0" ];
                then
                    for DOUBLE_DOUBLE_CONV in 1 0
                    do
                        echo "Filters:" $FILTERS "Batch norm:" $BATCH_NORM "Conv Transpose:" $CONV_TRANSPOSE "Squeeze:" $SQUEEZE "DD Conv:" $DOUBLE_DOUBLE_CONV
                        python3 train.py -p $CONFIG_PATH -f $FILTERS --batch_norm $BATCH_NORM --conv_transpose $CONV_TRANSPOSE --squeeze $SQUEEZE --double_double_conv $DOUBLE_DOUBLE_CONV
                    done
                else
                    echo "Filters:" $FILTERS "Batch norm:" $BATCH_NORM "Conv Transpose:" $CONV_TRANSPOSE "Squeeze:" $SQUEEZE
                    python3 train.py -p $CONFIG_PATH -f $FILTERS --batch_norm $BATCH_NORM --conv_transpose $CONV_TRANSPOSE --squeeze $SQUEEZE
                fi
            done
        done
    done
done

rm model_summary.txt output_frame.png *confusion_matrix.png
