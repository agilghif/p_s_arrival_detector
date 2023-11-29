"""This module defines a BentoML service that uses a Keras model to classify
digits.
"""
from datetime import datetime, timedelta
from types import NoneType
from typing import Optional, List, Tuple, Dict, Set
import math

import numpy as np
import bentoml
from pathlib import Path
import json

import redis
from bentoml.io import NumpyNdarray
from bentoml.io import JSON, Text
from pydantic import BaseModel

from pipeline import Pipeline, PipelineHasNotBeenInitializedException
from pydantic_models import *
import settings

# Model tags
BENTO_MODEL_TAG_P = "p_model:latest"
BENTO_MODEL_TAG_S = "s_model:latest"
BENTO_MODEL_TAG_MAG = "mag_model:latest"
BENTO_MODEL_TAG_DIST = "dist_model:latest"

# Runners
p_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_P).to_runner()
s_detector_runner = bentoml.keras.get(BENTO_MODEL_TAG_S).to_runner()
mag_runner = bentoml.keras.get(BENTO_MODEL_TAG_MAG).to_runner()
dist_runner = bentoml.keras.get(BENTO_MODEL_TAG_DIST).to_runner()

# Service
wave_arrival_detector = bentoml.Service("wave_arrival_detector", runners=[p_detector_runner, s_detector_runner, mag_runner, dist_runner])

# Pipelines
pipelines: Dict[str, Pipeline] = dict()

# Redis
redis_client = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=settings.REDIS_DB_NUM)

# Restart service ------------------------------------------------------------------------------------------------------
@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataRegister), output=Text())
def restart(input_data: json) -> str:
    # Get station code
    station_code: str = input_data.station_code

    # Insert station code
    # station_list: str
    # try:
    #     station_list = redis_client.get(settings.REDIS_STATION_LIST_NAME).decode("UTF8")
    # except AssertionError:
    #     redis_client.set(settings.REDIS_STATION_LIST_NAME, "")
    #     station_list = ""
    # station_list: Set = set(station_list.split("~"))
    # station_list.add(station_code)

    # Save station list name
    # redis_client.set(settings.REDIS_STATION_LIST_NAME, "~".join(station_list))

    # Create initial state
    redis_client.set(f"{station_code}~data_p", f"{0}~{datetime.min.strftime(settings.DATETIME_FORMAT)}~0")
    redis_client.set(f"{station_code}~data_s", f"{0}~{datetime.min.strftime(settings.DATETIME_FORMAT)}~0")
    redis_client.set(f"{station_code}~timer_s", datetime.min.strftime(settings.DATETIME_FORMAT))

    return "OK"


# Wave detection service -----------------------------------------------------------------------------------------------
@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataInference), output=JSON(pydantic_model=OutputDataInference))
def predict(input_data: json) -> json:
    # Unpack json
    x: np.ndarray = np.array(input_data.x)
    begin_time: datetime = datetime.strptime(input_data.begin_time, settings.DATETIME_FORMAT)
    station_code: str = input_data.station_code

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
    s_arrival_detected: bool = False
    s_arrival_time: datetime = ""
    s_arr_id = None
    new_s_event = False
    # prediction_s = np.array([])
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
        # "p_pred": prediction_p.tolist(),

        # S wave data
        "s_arr": s_arrival_detected,
        "s_arr_time": s_arrival_time.strftime(settings.DATETIME_FORMAT),
        "s_arr_id": f"{station_code}~{s_arr_id}",
        "new_s_event": new_s_event,
        # "s_pred": prediction_s.tolist()
    }

    return json.dumps(output)

# Earthquake magnitude & distance service ------------------------------------------------------------------------------
@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataMagLoc), output=JSON(pydantic_model=OutputDataMagLoc))
def approx_earthquake_statistics(input_data: json) -> json:
    x: np.ndarray = np.array([input_data.x])
    station_code: str = input_data.station_code

    # Magnitude
    magnitude = float(mag_runner.predict.run(x)[0][0])

    # Distance
    distance = float(dist_runner.predict.run(x)[0][0])

    # Depth
    # depth = depth_runner.predict.run(x)[0]

    # Output
    output = {
        "magnitude": magnitude,
        "distance": distance,
        "depth": 0.0,
        "station_code": station_code
    }

    return json.dumps(output)


@wave_arrival_detector.api(input=JSON(pydantic_model=InputDataMagLogRecalc),
                           output=JSON(pydantic_model=OutputDataMagLocRecalc))
def recalculate(input_data: json) -> json:
    # Unpack json data
    magnitudes: np.ndarray = np.array(input_data.magnitudes)
    distances: np.ndarray = np.array(input_data.distances).astype(np.complex128)
    station_latitudes: np.ndarray = np.array(input_data.station_latitudes).astype(np.complex128)
    station_longitudes: np.ndarray = np.array(input_data.station_longitudes).astype(np.complex128)

    # Cache values
    station_latitudes_rad = station_latitudes / 180.0 * np.pi * 6371.0
    station_longitudes_rad = station_longitudes / 180.0 * np.pi * 6371.0

    # Recalculate magnitude
    magnitude = np.mean(magnitudes)

    # Recalculate location
    # TODO : This formula is only for flat euclidian R2 space,
    #  find another more precise formula for intersection of three spheres.
    points = []
    for i in range(len(station_latitudes)-1):
        for j in range(i+1, len(station_latitudes)):
            # distance between two stations
            R = haversine(station_latitudes[i], station_longitudes[i], station_latitudes[j], station_longitudes[j])

            # Radians position of two stations
            xi = station_latitudes_rad[i]
            yi = station_longitudes_rad[i]
            xj = station_latitudes_rad[j]
            yj = station_longitudes_rad[j]
            ri = distances[i]
            rj = distances[j]

            x_delta = 0.5 * np.sqrt(
                2 * (ri**2+rj**2)/R**2 - (ri**2-rj**2)**2/R**4 - 1
            ) * (yj-yi)

            y_delta = 0.5 * np.sqrt(
                2 * (ri**2+rj**2)/R**2 - (ri**2-rj**2)**2/R**4 - 1
            ) * (xi-xj)

            x_base = 0.5*(xi+xj) + (ri**2-rj**2)/(2*R**2) * (xj-xi)

            y_base = 0.5*(yi+yj) + (ri**2-rj**2)/(2*R**2) * (yj-yi)

            x_1 = x_base + x_delta
            x_2 = x_base - x_delta
            y_1 = y_base + y_delta
            y_2 = y_base - y_delta

            points.append(np.array([[x_1, y_1], [x_2, y_2]]))

    # Find points with the least variance
    triplets = []
    variances = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Generate triplets
                triplet: np.ndarray = np.array([points[0][i], points[1][j], points[2][k]])
                triplets.append(triplet)

                # Calculate variance
                variances.append(triplet.var(axis=0).sum())

    # Select the triplets with the least variance value
    variances = np.array(variances)
    argmin = variances.argmin()

    # Retrieve argmin-th triplet
    triplet: np.ndarray = triplets[argmin]

    # Project triplet into real number
    triplet = triplet.real

    # Take the average
    ans = triplet.mean(axis=0)

    # Convert result back to degree
    ans *= 180 / np.pi / 6371.0

    # Compose output
    output = {
        "station_codes": input_data.station_code,
        "magnitude": float(magnitude),
        "latitude": float(ans[0]),
        "longitude": float(ans[1]),
        "depth": 0.0
    }

    return json.dumps(output)



# AUXILIARY FUNCTIONS --------------------------------------------------------------------------------------------------
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
        mean_approx = first_detection_index - (window_size - round(np.mean(deviation)))

        return True, mean_approx, len(detected_indices)

    # Case if no p wave detected
    return False, 0.0, 0


def haversine(lat1, lon1, lat2, lon2):

    # Convert latitude and longitude from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Difference in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = 6371.0 * c

    return distance
