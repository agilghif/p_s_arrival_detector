from pydantic import BaseModel
from typing import List


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
    station_code: str


class InputDataMagLogRecalc(BaseModel):
    station_codes: List[str]
    station_latitudes: List[float]
    station_longitudes: List[float]
    magnitudes: List[float]
    distances: List[float]
    depths: List[float]


class OutputDataMagLocRecalc(BaseModel):
    station_codes: List[str]
    magnitude: float
    latitude: float
    longitude: float
    depths: float
