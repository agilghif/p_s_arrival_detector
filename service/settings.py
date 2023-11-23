from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Models
P_MODEL_PATH = Path("pipeline/model_p_best.pipeline")

# Parameters
P_THRESHOLD = 0.5
S_THRESHOLD = 0.5
WINDOW_SIZE = 382
SAMPLING_RATE = 20.0
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
EARTHQUAKE_PICK_TIME_THRESHOLD = timedelta(seconds=6)
S_WAVE_DETECTION_DURATION = timedelta(minutes=1)

# Redis client
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', "6379")
REDIS_DB_NUM = 0
REDIS_STATION_LIST_NAME = "station_set"
