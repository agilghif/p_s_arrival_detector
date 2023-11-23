import numpy as np
import math

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

def recalculate(input_data: dict) -> dict:
    # Unpack json data
    magnitudes: np.ndarray = np.array(input_data["magnitudes"])
    distances: np.ndarray = np.array(input_data["distances"]).astype(np.complex128)
    station_latitudes: np.ndarray = np.array(input_data["station_latitudes"]).astype(np.complex128)
    station_longitudes: np.ndarray = np.array(input_data["station_longitudes"]).astype(np.complex128)

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
        "station_codes": input_data["station_codes"],
        "magnitude": float(magnitude),
        "latitude": float(ans[0]),
        "longitude": float(ans[1]),
        "depth": 0.0
    }

    return output


input_data = {
    "station_codes": ["TNTI", "TOLI", "GANI"],
    "station_latitudes": [0.7718, 1.1214, -2.5927],
    "station_longitudes": [127.3667, 120.7944, 140.1678],
    "magnitudes": [6, 5.8, 6.1],
    "distances": [10, 21, 11],
    "depths": [10, 11, 12]
}

print(recalculate(input_data))