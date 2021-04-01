# Collect the IR data on Raspberry PI 4B

# Prepare operating system and software on Raspberry PI
- download and flash `Raspberry Pi OS` image to the SD-Card. See details on https://www.raspberrypi.org/software/
- expand the primary partition on the SD-Card (e.g. with gparted)
- (optional) enable ssh by running `touch ssh` on `boot` partition of the SD-Card
- connect IR camera with cables to GPIOs (seehttps://www.element14.com/community/servlet/JiveServlet/showImage/102-92640-8-726998/GPIO-Pi4.png )
- 3v3 (1st in bottom left row)
- I2C SDA (2nd in bottom left row)
- I2C SCL (3rd in bottom left row)
- GND (5th in bottom left row)

- start Raspberry and either connect via ssh or use display and keyboard connected to PI
- user `pi`, change password to `pivision`
- Run commands:
```
sudo apt update
sudo apt upgrade

# Install required packages
sudo apt-get install -y python-smbus i2c-tools
sudo apt-get install libatlas-base-dev

cd /media/pi/data
git clone xxxxx
cd ir_vision

# Prepare virtual environment
python3 -m venv venv
. venv/bin/activate
pip3 install -r data_collection/requirement.txt
```
- Enable I2C. Open `sudo nano /boot/config.txt` and:
  - change line `dtparam=i2c_arm=on` to `dtparam=i2c_arm=on,i2c_arm_baudrate=1000000`, 
  - add lines:
```
# Enable optical camera
start_x=1             # essential
gpu_mem=128           # at least, or maybe more if you wish
disable_camera_led=1  # optional, if you don't want the led to glow
```
- then reboot
- check if camera can be detected on I2c: `sudo i2cdetect -y 1` at address 0x33  
- for more details see: 
    - https://learn.adafruit.com/adafruit-mlx90640-ir-thermal-camera/python-circuitpython
    - https://makersportal.com/blog/2020/6/8/high-resolution-thermal-camera-with-raspberry-pi-and-mlx90640

- try if the IR camera works - `python ./data_collection/misc/manual_ir_test.py` (one should see preview from IR camera plotted in matplotlib)
- try normal camera - `raspistill -t 0` 
- (optional) mount pi storage ib your PC for convenience
```
mkdir tmp; cd tmp; mkdir pi_mnt; sshfs pi@10.11.12.76:/home/pi/Projects pi_mnt
```

# Run the recording manually
```
. venv/bin/activate
cd data_collection/src
python main.py

# visit 0.0.0.0:8888 in a browser to see live status
```

# Or use as service (starts as daemon):
  Copy service file one time and enable service:
```
sudo cp ir_vision.service  /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ir_vision 
```
- And then use as a service with standard commands:
```
sudo systemctl stop ir_vision
sudo systemctl start ir_vision
sudo systemctl status ir_vision
```
