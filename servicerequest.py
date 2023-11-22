import os
from typing import Tuple
import json

import numpy as np
import requests
import pickle
import matplotlib.pyplot as plt
import datetime

SERVICE_URL_PREDICT = "http://localhost:3000/predict"
SERVICE_URL_RESTART = "http://localhost:3000/restart"
SERVICE_URL_MAG_LOC = "http://localhost:3000/approx_earthquake_statistics"

def make_request_to_bento_service(service_url: str, input_array: np.ndarray, begin_time: datetime.datetime,
                                  station_code: str) -> str:
    input_data = {'x': input_array.tolist(),
                  # 'begin_time': begin_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
                  'station_code': station_code}
    serialized_input_data = json.dumps(input_data)
    print("sending response...")
    response = requests.post(
        service_url,
        data=serialized_input_data,
        headers={"content-type": "application/json"}
    )
    print("response sent...")
    print(response.text)
    return json.loads(response.text)

def decimate(earthquake):
    indices = [int(num) for num in np.linspace(0, len(earthquake), int(len(earthquake) / 5), endpoint=False)]
    return earthquake[indices]

def main():
    station_code = "AGL"
    print(f"Registering station {station_code}")
    requests.post(SERVICE_URL_RESTART,
                  data=json.dumps({'station_code': station_code}),
                  headers={"content-type": "application/json"})

    print("Starting making requests...")
    time = datetime.datetime.now()
    for filename in os.listdir("./sample_data"):
        with open(f"./sample_data/{filename}", "rb") as f:
            earthquake = np.load(f)

            # Misalkan setiap satu detik, service lain ngirim 500 datapoints (bisa aja 20, tapi ini exaggerated)
            delta = datetime.timedelta(seconds=1)
            last_earthquake_id = station_code + "~0"
            for i in range(0, len(earthquake), 500):
                earthquake_data = earthquake[i:i+500]
                response: dict = json.loads(make_request_to_bento_service(
                    SERVICE_URL_MAG_LOC,
                    earthquake_data,
                    time,
                    station_code
                ))

                fig, axs = plt.subplots(3)
                axs[0].plot(earthquake_data)
                axs[0].set_xlim(0,500)

                p_pred = response["p_pred"]
                s_pred = response["s_pred"]

                if response["p_arr"] and response["new_p_event"]:
                    p_arrival_time = datetime.datetime.strptime(response["p_arr_time"], "%Y-%m-%d %H:%M:%S.%f")
                    idx = round((p_arrival_time - time).total_seconds() * 20)
                    print(idx, p_arrival_time, time)
                    axs[0].vlines(idx, ymin=np.min(earthquake_data), ymax=np.max(earthquake_data), linestyles="dashed", colors="red")
                    axs[1].imshow(np.array(p_pred).squeeze().T, aspect='auto', vmin=0, vmax=1)

                if response["s_arr"] and response["new_s_event"]:
                    s_arrival_time = datetime.datetime.strptime(response["s_arr_time"], "%Y-%m-%d %H:%M:%S.%f")
                    idx = round((s_arrival_time - time).total_seconds() * 20)
                    print(idx, s_arrival_time, time)
                    axs[0].vlines(idx, ymin=np.min(earthquake_data), ymax=np.max(earthquake_data), linestyles="dashed", colors="blue")
                    if s_pred is not None:
                        axs[2].imshow(np.array(s_pred).squeeze().T, aspect='auto', vmin=0, vmax=1)

                axs[0].set_title(f"Gempa {filename}[{i}:{i+500}]")
                plt.show()

                time += delta * len(earthquake_data) / 20


if __name__ == "__main__":
    main()
