import xml.etree.ElementTree as ET
from pathlib import Path
import sys

# TODO: MOve this file to parent folder

# config.xml lives in the parent folder of the folder of our script.
XML_FILE = Path(__file__).parent / "config.xml"

try:
    tree = ET.parse(XML_FILE)
    root = tree.getroot()

    INPUT_DIRS = [
        Path(p.text).resolve() for p in root.findall("directories/inputs/path")
    ]
    OUTPUT_DIRS = [
        Path(p.text).resolve() for p in root.findall("directories/outputs/path")
    ]
    SCALER_DIRS = [
        Path(p.text).resolve() for p in root.findall("directories/scalers/path")
    ]

    BUFFER_DIR = Path(root.findtext("directories/buffer_dir"))

    MODEL_DIR_EDGE = Path(root.findtext("directories/model_dir_edge"))
    MODEL_DIR_GPU = Path(root.findtext("directories/model_dir_gpu"))
    MODEL_DIR_CLASSIFY = Path(root.findtext("directories/model_dir_classify"))

    SAMPLING_RATE = int(root.findtext("audio_params/sampling_rate"))
    N_MELS = int(root.findtext("audio_params/n_mels"))
    FIXED_WIDTH = int(root.findtext("audio_params/fixed_width"))
    HOP_LENGTH = int(root.findtext("audio_params/hop_length"))
    N_FFT = int(root.findtext("audio_params/n_fft"))

except FileNotFoundError:
    print(f"CRITICAL ERROR: Configuration file '{XML_FILE.name}' not found.")
    sys.exit(1)
except AttributeError as e:
    print(f"CRITICAL ERROR: Missing a required tag in your XML config. {e}")
    sys.exit(1)
except ValueError as e:
    print(f"CRITICAL ERROR: Could not convert XML audio parameters to integers. {e}")
    sys.exit(1)
