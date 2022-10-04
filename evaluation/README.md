# Hardware evaluation

The model evaluation on Thermo Presence dataset is available using [`evaluate.py`](./evaluate.py) script. The script is parametrized and possible options are described below.

```console
Options:
  --inference_type [tflite|edgetpu|myriad]
                                  Inference framework (device)
  --model_path TEXT               Path to model
  --data_path TEXT                Path to HDF files
  --help                          Show this message and exit.
```

Script was utilized to benchmark [Raspberry Pi 4B](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/) with [Intel Neural Compute Stick 2](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html) and [Google Coral USB Accelerator](https://coral.ai/products/accelerator/). The achieved results are described in the chapter [Thermo Presence: The Low-resolution Thermal Image Dataset and Occupancy Detection Using Edge Devices](https://wydawnictwo.umg.edu.pl/pp-rai2022/pdfs/11_pp-rai-2022-094.pdf).

```
@inproceedings{thermo-presence-pp-rai,
  author    = "Aszkowski, Przemysław and Piechocki, Mateusz",
  title     = "Thermo Presence: The Low-resolution Thermal Image Dataset and Occupancy Detection Using Edge Devices",
  booktitle = "Proceedings of the 3rd Polish Conference on Artificial Intelligence",
  year      = 2022,
  pages     = "49--52",
  publisher = "House of Gdynia Maritime University",
  address   = "Gdynia, Poland"
}
```
