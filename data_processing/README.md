# Set of tools for labeling and processing data

Tested with Python 3.7.5 on Linux Ubuntu 18.04

## Setup venv and install requirements:
```
python3 -m venv venv
. venv/bin/activate
pip install -r data_processing/requirements.txt
```

## Run data labeling tool:
```
cd data_processing
export PYTHONPATH=$PYTHONPATH:`pwd` && python3 data_labeling/data_labeling_main.py
```

## Or run notebooks:
``` 
jupyter notebook
```
and navigate to the main notebook (notebooks/model_training_and_test.ipynb) to train and validate the model.
