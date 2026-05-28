from pathlib import Path

DATA_DIR      = Path(r'C:\pavimentos\data\raw')
YOLO_DATA_DIR = Path(r'C:\pavimentos\data\processed\yolo_dataset')
MODEL_DIR     = Path(r'C:\pavimentos\models')
RESULTS_DIR   = Path(r'C:\pavimentos\results')

GDINO_CKPT   = Path(r'C:\pavimentos\models\groundingdino_swint_ogc.pth')
GDINO_CONFIG = Path(r'C:\pavimentos\models\GroundingDINO_SwinT_OGC.py')

ROBOFLOW_API_KEY = ''
ROBOFLOW_WS      = ''
ROBOFLOW_PROJECT = ''
ROBOFLOW_VERSION = 1

CLASS_NAMES = [
    'buraco',
    'remendo',
    'trinca',
    'sinalizacao_vertical',
    'sinalizacao_horizontal',
    'drenagem',
    'vegetacao',
]
NUM_CLASSES = len(CLASS_NAMES)

YOLO_MODEL  = 'yolov8m.pt'
IMG_SIZE    = 640
BATCH_SIZE  = 8
EPOCHS      = 50
CONF_THRESH = 0.25
IOU_THRESH  = 0.45
SEED        = 42

BOX_THRESHOLD  = 0.15
TEXT_THRESHOLD = 0.10
NMS_THRESHOLD  = 0.50
MIN_BOX_AREA   = 0.0005
RESIZE_WIDTH   = 1280
RESIZE_HEIGHT  = 720

TEXT_PROMPTS = {
    'buraco'                 : 'pothole',
    'remendo'                : 'asphalt patch',
    'trinca'                 : 'road crack',
    'sinalizacao_vertical'   : 'traffic sign',
    'sinalizacao_horizontal' : 'lane marking',
    'drenagem'               : 'drainage',
    'vegetacao'              : 'vegetation',
}

ICM_PESO = {
    'buraco'                 : 5,
    'remendo'                : 2,
    'trinca'                 : 3,
    'sinalizacao_vertical'   : 1,
    'sinalizacao_horizontal' : 1,
    'drenagem'               : 2,
    'vegetacao'              : 1,
}
