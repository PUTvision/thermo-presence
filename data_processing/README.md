- Tested with Python 3.7.5 on Linux Ubuntu 18.04

- Setup venv and install requirements:
```
python3 -m venv venv
pip install -r data_processing/requirements.txt
```

- Run data labeling tool:
```
cd data_processing
export PYTHONPATH=$PYTHONPATH:`pwd` && python3 data_labeling/data_labeling_main.py
```

- Or run notebooks:
``` 
jupyter notebook
```
