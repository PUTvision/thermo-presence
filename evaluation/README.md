# Hardware evaluation


The model evaluation on Thermo Presence dataset is available using `evaluate.py` script. The script is parametrized and possible options are described below.

```console
Options:
  --inference_type [tflite|edgetpu|myriad]
                                  Inference framework (device)
  --model_path TEXT               Path to model
  --validation_input TEXT         Path to validation input file
  --validation_output TEXT        Path to validation output file
  --help                          Show this message and exit.
```

Script was utilized to benchmark [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) with [Intel Neural Compute Stick 2](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) and [Google Coral USB Accelerator](https://coral.ai/products/accelerator/).
