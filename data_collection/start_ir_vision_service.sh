#!/usr/bin/env bash

echo $BASHPID > /tmp/ir_vision.pid

cd /home/pi/Projects/ir_vision
. venv/bin/activate
cd data_collection/src
python main.py

exit -1
