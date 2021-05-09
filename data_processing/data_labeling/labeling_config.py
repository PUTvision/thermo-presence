import getpass
import os

import cv2


ROOT_DATA_DIR_PATH = ''  # directory where the raw data (recorded on Raspberry) and available on OwnCloud are stored

if getpass.getuser() == 'przemek':
    # ROOT_DATA_DIR_PATH = '/media/przemek/data/ir_from_owncloud/'
    ROOT_DATA_DIR_PATH = '/media/data/temporary/thermo-presence/'


if not ROOT_DATA_DIR_PATH:
    raise Exception("Please specify root path directory of data files (ROOT_DATA_DIR_PATH in labeling_config.py)!")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_LABELS_DIR = os.path.join(SCRIPT_DIR, 'labeling_output')


MIN_TEMPERATURE_ON_PLOT = 20  # None for auto range
MAX_TEMPERATURE_ON_PLOT = 30  # None for auto range
IR_FRAME_INTERPOLATION_METHOD = cv2.INTER_CUBIC
IR_FRAME_RESIZE_MULTIPLIER = 16
RGB_FRAME_RESIZE_MULTIPLIER = 2


SUBDIRECTORIES_TO_ANNOTATE = [
    # Room A sequence 1
    '31_03_21__318__3or4_people/1/006__11_44_59',
    '31_03_21__318__3or4_people/1/007__11_48_59',
    '31_03_21__318__3or4_people/1/008__11_52_59',
    '31_03_21__318__3or4_people/1/009__11_57_00',

    # Room A sequence 2
    '31_03_21__318__3or4_people/2/000__14_15_19',
    '31_03_21__318__3or4_people/2/001__14_19_19',
    '31_03_21__318__3or4_people/2/002__14_23_19',
    '31_03_21__318__3or4_people/2/003__14_27_20',
    '31_03_21__318__3or4_people/2/004__14_31_20',
    '31_03_21__318__3or4_people/2/005__14_35_20',
    '31_03_21__318__3or4_people/2/006__14_39_20',
    '31_03_21__318__3or4_people/2/007__14_43_20',
    '31_03_21__318__3or4_people/2/008__14_47_20',
    '31_03_21__318__3or4_people/2/009__14_51_20',
    '31_03_21__318__3or4_people/2/010__14_55_20',
    '31_03_21__318__3or4_people/2/011__14_59_20',
    '31_03_21__318__3or4_people/2/012__15_03_21',
    '31_03_21__318__3or4_people/2/013__15_07_21',
    '31_03_21__318__3or4_people/2/014__15_11_21',
    '31_03_21__318__3or4_people/2/015__15_15_21',
    '31_03_21__318__3or4_people/2/016__15_19_21',
    # '31_03_21__318__3or4_people/2/017__15_23_21',
    # '31_03_21__318__3or4_people/2/018__15_27_21',
    # '31_03_21__318__3or4_people/2/019__15_31_21',
    # '31_03_21__318__3or4_people/2/020__15_35_21',
    # '31_03_21__318__3or4_people/2/021__15_39_21',
    # '31_03_21__318__3or4_people/2/022__15_43_22',
    # '31_03_21__318__3or4_people/2/023__15_47_22',
    # '31_03_21__318__3or4_people/2/024__15_51_22',
    # '31_03_21__318__3or4_people/2/025__15_55_22',

    # Room B sequence 1
    # '05_05_2021__0to5_people/000__11_06_13',  # skipped, moving camera
    # '05_05_2021__0to5_people/001__12_58_19',  # skipped, moving camera
    # '05_05_2021__0to5_people/002__13_02_19',  # something strange with the camera
    # '05_05_2021__0to5_people/003__13_06_19',  # something strange with the camera
    '05_05_2021__0to5_people/004__13_10_20',
    # '05_05_2021__0to5_people/005__13_14_20',  # something strange with the camera
    # '05_05_2021__0to5_people/006__13_18_20',  # something strange with the camera
    '05_05_2021__0to5_people/007__13_22_20',
    '05_05_2021__0to5_people/008__13_26_20',

    # Room C sequence 1
    # '05_05_2021__0to5_people/009__13_30_20',  # skipped, moving camera
    # '05_05_2021__0to5_people/010__13_34_20',  # skipped, moving camera
    '05_05_2021__0to5_people/011__13_38_20',
    '05_05_2021__0to5_people/012__13_42_20',
    '05_05_2021__0to5_people/013__13_46_21',
    '05_05_2021__0to5_people/014__13_50_21',
    '05_05_2021__0to5_people/015__13_54_21',
    # '05_05_2021__0to5_people/016__13_58_21',  # skipped, moving camera

]


