"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""
from datetime import datetime, timedelta
from types import NoneType
from typing import Optional, List, Tuple, Dict, Set

import numpy as np
import bentoml
from pathlib import Path
import json

import redis
from bentoml.io import NumpyNdarray
from bentoml.io import JSON, Text
from pydantic import BaseModel

from pipeline import Pipeline, PipelineHasNotBeenInitializedException
import settings

BENTO_MODEL_TAG_P = "p_model:latest"
BENTO_MODEL_TAG_S = "s_model:latest"
BENTO_MODEL_TAG_MAG = "mag_model:latest"
BENTO_MODEL_TAG_DIST = "dist_model:latest"

p_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_P).to_runner()
s_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_S).to_runner()
mag_runner = bentoml.keras.get(BENTO_MODEL_TAG_MAG).to_runner()
dist_runner = bentoml.keras.get(BENTO_MODEL_TAG_DIST).to_runner()

wave_arrival_detector = bentoml.Service("wave_arrival_detector", runners=[p_detector_runner, s_detector_runner])
magnitude_distance_calculator = bentoml.Service("magnitude_location", runners=[mag_runner, dist_runner])

# Setting pipeline data
pipeline_p = Pipeline(settings.P_MODEL_PATH, settings.WINDOW_SIZE)
pipelines: Dict[str, Pipeline] = dict()

# Redis client to access last earthquake info
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB_NUM)


# RESTART JSON STRUCTURE
class InputDataRegister(BaseModel):
    station_code: str


# P/S WAVE DETECTION JSON STRUCTURE
class InputDataInference(BaseModel):
    x: List[List[float | int]]
    begin_time: str  # should be like settings.DATETIME_FORMAT format
    station_code: str


class OutputDataInference(BaseModel):
    station_code: str
    init_end: bool

    # P wave data
    p_arr: bool
    p_arr_time: str
    p_arr_id: int
    new_p_event: bool

    # S wave data
    s_arr: bool
    s_arr_time: str
    s_arr_id: int
    new_s_event: bool

# MAGNITUDE / LOCATION JSON STRUCTURE
class InputDataMagLoc(BaseModel):
    x: List[List[int | float]]
    station_code: str

class OutputDataMagLoc(BaseModel):
    magnitude: float
    distance: float
    depth: float

@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataRegister), output=Text())
def restart(input_data: json) -> str:
    # Get station code
    station_code: str = input_data.station_code

    # Insert station code
    station_list: str
    try:
        station_list = redis_client.get(settings.REDIS_STATION_LIST_NAME).decode("UTF8")
    except AssertionError:
        redis_client.set(settings.REDIS_STATION_LIST_NAME, "")
        station_list = ""
    station_list: Set = set(station_list.split("~"))
    station_list.add(station_code)

    # Save station list name
    redis_client.set(settings.REDIS_STATION_LIST_NAME, "~".join(station_list))

    # Create initial state
    redis_client.set(f"{station_code}~data_p", f"{0}~{datetime.min.strftime(settings.DATETIME_FORMAT)}~0")
    redis_client.set(f"{station_code}~data_s", f"{0}~{datetime.min.strftime(settings.DATETIME_FORMAT)}~0")
    redis_client.set(f"{station_code}~timer_s", datetime.min.strftime(settings.DATETIME_FORMAT))

    return "OK"

# P WAVE DETECTION SERVICES
@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataInference), output=JSON(pydantic_model=OutputDataInference))
def predict(input_data: json) -> json:
    # Unpack json
    x: np.ndarray = np.array(input_data.x)
    begin_time: datetime = datetime.strptime(input_data.begin_time, settings.DATETIME_FORMAT)
    station_code: str = input_data.station_code
    pipeline: Pipeline
    if station_code in pipelines:
        pipeline = pipelines[station_code]
    else:
        pipeline = Pipeline(settings.P_MODEL_PATH, settings.WINDOW_SIZE, name=station_code)
        pipelines[station_code] = pipeline

    # Preprocess x
    x: np.ndarray
    try:
        x = pipeline.process(x)
    except PipelineHasNotBeenInitializedException:
        output = {
            "station_code": station_code,
            "init_end": False,

            # P wave data
            "p_arr": False,
            "p_arr_time": "",
            "p_arr_id": "",
            "new_p_event": False,

            # S wave data
            "s_arr": False,
            "s_arr_time": "",
            "s_arr_id": "",
            "new_s_event": False,
        }
        return json.dumps(output)

    # ### P WAVE DETECTION ###
    # Make prediction
    prediction_p: np.ndarray = p_detector_runner.predict.run(x)

    # Extract insight
    p_arrival_detected, p_arrival_time, p_arr_id, new_p_event = \
        examine_prediction(prediction_p, station_code, begin_time, is_p=True)

    # ### S WAVE DETECTION ###
    # Make s prediction if p wave is detected
    s_arrival_detected: bool = False
    s_arrival_time: datetime = ""
    s_arr_id = None
    new_s_event = False
    timer_s: datetime = datetime.strptime(
        redis_client.get(f"{station_code}~timer_s").decode("UTF8"), settings.DATETIME_FORMAT)

    if p_arrival_detected and new_p_event:
        # Create an s timer if new event detected
        timer_s = p_arrival_time
        redis_client.set(f"{station_code}~timer_s", p_arrival_time.strftime(settings.DATETIME_FORMAT))

    if timer_s - begin_time <= settings.S_WAVE_DETECTION_DURATION:
        # Make prediction
        prediction_s: np.ndarray = s_detector_runner.predict.run(x)

        # Extract insight
        s_arrival_detected, s_arrival_time, s_arr_id, new_s_event = \
            examine_prediction(prediction_s, station_code, begin_time, is_p=False)

    # ### SUMMARIZE ###
    output = {
        "station_code": station_code,
        "init_end": True,

        # P wave data
        "p_arr": p_arrival_detected,
        "p_arr_time": p_arrival_time.strftime(settings.DATETIME_FORMAT),
        "p_arr_id": f"{station_code}~{p_arr_id}",
        "new_p_event": new_p_event,

        # S wave data
        "s_arr": s_arrival_detected,
        "s_arr_time": s_arrival_time.strftime(settings.DATETIME_FORMAT),
        "s_arr_id": f"{station_code}~{s_arr_id}",
        "new_s_event": new_s_event,
    }

    return json.dumps(output)

@magnitude_distance_calculator.api(input=JSON(pydantic_model=InputDataInference), output=JSON(pydantic_model=OutputDataInference))
def approx_earthquake_statistics(input_data: json) -> json:
    x: np.ndarray = np.array(input_data.x)
    station_code: str = input_data.station_code

    # Magnitude
    magnitude = mag_runner.predict.run(x)

    # Distance
    distance = dist_runner.predict.run(x)

    # Depth
    # depth = depth_runner.predict.run(x)

    # Output
    output = {
        "magnitude": magnitude,
        "distance": distance,
        "depth": 0.0
    }

    return json.dumps(output)


# AUXILIARY FUNCTIONS
def examine_prediction(prediction: np.ndarray, station_code: str, begin_time: datetime, is_p: bool)\
        -> Tuple[bool, datetime, int, bool]:
    """Examine the prediction result, returns """
    # Check for wave arrival
    arrival_detected, arrival_pick_idx, arrival_count = pick_arrival(prediction, threshold=settings.P_THRESHOLD)

    # Present result
    # -- Convert p_arrival_idx to timestamps
    arrival_time = begin_time + timedelta(seconds=arrival_pick_idx/settings.SAMPLING_RATE)

    # Check last earthquake occurrence, note that le = last earthquake
    le_id, le_time, le_count = redis_client.get(f"{station_code}~data_{'p' if is_p else 's'}").decode("UTF-8").split("~")
    le_id: int = int(le_id)
    le_time: datetime = datetime.strptime(le_time, settings.DATETIME_FORMAT)
    le_count: int = int(le_count)

    is_new_earthquake = False
    if arrival_detected:
        # Case if detected earthquake is continuation from previous inference
        if abs(le_time - arrival_time) < settings.EARTHQUAKE_PICK_TIME_THRESHOLD:
            # refine pick time calculation
            arrival_time = le_time + (arrival_time - le_time) * arrival_count/(le_count + arrival_count)
            arrival_count += le_count

        # Case if detected earthquake is a new event (not a continuation from previous inference)
        else:
            is_new_earthquake = True
            le_id += 1

        # Save state
        redis_client.set(f"{station_code}~data_{'p' if is_p else 's'}", f"{le_id}~{str(arrival_time)}~{arrival_count}")

    return arrival_detected, arrival_time, le_id, is_new_earthquake


def pick_arrival(prediction: np.ndarray, threshold=0.5, window_size=settings.WINDOW_SIZE) -> Tuple[bool, float, int]:
    """retrieve the existence of p wave, its pick location, and #detection in prediction from given prediction result"""
    # Detect p wave occurrence
    detected_indices = np.where((prediction > threshold).any(axis=1))[0]  # Index where p wave arrival is detected

    # Case if p wave is detected
    if detected_indices.any():
        first_detection_index = detected_indices[0]
        ideal_deviation = np.array(
            detected_indices) - first_detection_index  # Location of p wave arrival ideally follows # this value

        # For all triggered windows, find its argmax
        argmax = np.array(prediction[detected_indices].argmax(axis=1))  # p wave pick index in every windows
        deviation = argmax + ideal_deviation  # predicted deviation

        # Find mean while excluding outliers
        mean_approx = first_detection_index - (window_size - round(mean_without_outliers(deviation)))

        return True, mean_approx, len(detected_indices)

    # Case if no p wave detected
    return False, 0.0, 0


def mean_without_outliers(arr, threshold=3.0):
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    mask = np.abs(arr - median) / mad < threshold
    return np.mean(arr[mask])