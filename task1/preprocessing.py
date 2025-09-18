import os
from collections import defaultdict
from datetime import datetime
from logging import getLogger
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from geopy.location import Location
from sklearn.preprocessing import FunctionTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

root_path = Path(os.getcwd()).parent
dataset_filename = "houses_Madrid.csv"
save_path = root_path / "data" / dataset_filename
DATASET_SCHEME = defaultdict(list)
logger = getLogger()

NEIGHBORHOOD_FEATURE_NAMES = [
    "sq_mt_price",
    "center_distance",
    "bearing_sin",
    "bearing_cos",
    "latitude",
    "longitude",
]
NEIGHBORHOOD_INDICES = [
    *NEIGHBORHOOD_FEATURE_NAMES,
    "neighborhood_name",
    "district_name",
]
RESTORE_PRICE_NEIGHBORS = 3
RETRY_WAIT = 2
STOP_AFTER_ATTEMPT = 10
LOCATOR_TIMEOUT = 10


ALL_BOOLEAN_FEATURES = [
    "is_floor_under",
    "has_central_heating",
    "has_individual_heating",
    "are_pets_allowed",
    "has_ac",
    "is_new_development",
    "is_renewal_needed",
    "has_fitted_wardrobes",
    "has_lift",
    "is_exterior",
    "has_garden",
    "has_pool",
    "has_terrace",
    "has_balcony",
    "has_storage_room",
    "is_furnished",
    "is_kitchen_equipped",
    "is_accessible",
    "has_green_zones",
    "has_parking",
    "has_private_parking",
    "has_public_parking",
    "is_parking_included_in_price",
    "is_orientation_north",
    "is_orientation_west",
    "is_orientation_south",
    "is_orientation_east",
    "is_orientation_stated",
]
COMPLETE_BOOLEAN_FEAURES = ["is_renewal_needed", "has_parking"]
EMPTY_BOOLEAN_FEATURES = [
    "is_kitchen_equipped",
    "is_furnished",
    "are_pets_allowed",
    "has_public_parking",
    "has_private_parking",
]
POLAR_BOOLEAN_FEATURES = [
    "has_ac",
    "has_terrace",
    "is_parking_included_in_price",
    "has_storage_room",
    "has_pool",
    "is_accessible",
    "has_green_zones",
    "has_balcony",
    "has_garden",
    "has_individual_heating",
    "has_central_heating",
    "has_lift",
    "is_new_development",
    "is_floor_under",
    "is_exterior",
    "has_fitted_wardrobes",
]
INCOMPLETE_BOOLEAN_FEATURES = [
    "is_orientation_south",
    "is_orientation_north",
    "is_orientation_east",
    "is_orientation_west",
]
NON_EMPTY_FEATURES = list(
    set(ALL_BOOLEAN_FEATURES).difference(set(EMPTY_BOOLEAN_FEATURES))
)

ENERGY_CERTIFICATE_ORDER = [
    "no indicado",
    "en trámite",
    "G",
    "F",
    "E",
    "D",
    "C",
    "B",
    "A",
    "inmueble exento",
]
HOUSE_TYPE_ORDER = [
    np.nan,
    "HouseType 1: Pisos",
    "HouseType 5: Áticos",
    "HouseType 4: Dúplex",
    "HouseType 2: Casa o chalet",
]


if save_path.exists():
    df = pd.read_csv(save_path)
else:
    download_dir = (
        Path(kagglehub.dataset_download("mirbektoktogaraev/madrid-real-estate-market"))
        / dataset_filename
    )
    df = pd.read_csv(download_dir, index_col="id")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df.to_csv(save_path)


@FunctionTransformer
def convert_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Applies target transformations.

    Applies log transform to the target, since it was
    shown o exhibit lognormal distribution earlier.

    Parameters
    ----------
    frame : pd.DataFrame
        initial / previous step frame

    Returns
    -------
    pd.DataFrame
        frame with changed target
    """

    frame["buy_price"] = np.log(frame["buy_price"])
    return frame


@FunctionTransformer
def process_square_meters(frame: pd.DataFrame) -> pd.DataFrame:
    """Process square meters columns.

    - Mutually restores 'sq_mt_built' and 'sq_mt_useful'. Some entries are deleted
    if no meaningful value is provided for either of the columns.
    - Applies log transform to the derived features 'sq_mt_built_proc' and 'sq_mt_useful_proc'.
    - Adds features 'sq_mt_built_present' and 'sq_mt_useful_present' of the presence for
    'sq_mt_built' and 'sq_mt_useful' respectfully.

    Parameters
    ----------
    frame : pd.DataFrame
        initial / previous step frame

    Returns
    -------
    pd.DataFrame
        frame with new features
    """

    # Derive portion of the dataset, where both features are present
    minimum_sq_mt = frame["n_rooms"].fillna(0) + frame["n_bathrooms"].fillna(0)
    sq_mt_built_present = ~frame["sq_mt_built"].isna()
    sq_mt_useful_present = ~frame["sq_mt_useful"].isna()
    sq_mt_built_meaningful = frame["sq_mt_built"] > minimum_sq_mt
    sq_mt_useful_meaningful = frame["sq_mt_useful"] > minimum_sq_mt
    clean_frame = frame[["sq_mt_built", "sq_mt_useful", "n_rooms", "n_bathrooms"]][
        sq_mt_built_present
        & sq_mt_useful_present
        & sq_mt_built_meaningful
        & sq_mt_useful_meaningful
    ][["sq_mt_built", "sq_mt_useful"]]

    # Derive median ratio of built/useful square meters and restore both columns
    median_ratio = (clean_frame["sq_mt_built"] / clean_frame["sq_mt_useful"]).median()

    frame = frame[
        (sq_mt_built_present & sq_mt_built_meaningful)
        | (sq_mt_useful_present & sq_mt_useful_meaningful)
    ]

    frame["sq_mt_built_proc"] = frame["sq_mt_built"]
    frame["sq_mt_built_proc"].loc[frame["sq_mt_built_proc"].isna()] = (
        frame["sq_mt_useful"].loc[frame["sq_mt_built_proc"].isna()] * median_ratio
    )

    frame["sq_mt_useful_proc"] = frame["sq_mt_useful"]
    frame["sq_mt_useful_proc"].loc[frame["sq_mt_useful_proc"].isna()] = (
        frame["sq_mt_built"].loc[frame["sq_mt_useful_proc"].isna()] / median_ratio
    )

    # Apply log transform to new features
    frame["sq_mt_built_proc"] = np.log(frame["sq_mt_built_proc"])
    frame["sq_mt_useful_proc"] = np.log(frame["sq_mt_useful_proc"])

    # Add square meters numbers' presence
    frame["sq_mt_built_present"] = sq_mt_built_present & sq_mt_built_meaningful
    frame["sq_mt_useful_present"] = sq_mt_useful_present & sq_mt_useful_meaningful

    return frame


@retry(wait=wait_fixed(RETRY_WAIT), stop=stop_after_attempt(STOP_AFTER_ATTEMPT))
def _geocode_retry(locator: Nominatim, query: str) -> Location:
    return locator.geocode(query)


def get_distance_and_bearing(
    latitude1, longitude1, latitude2, longitude2
) -> tuple[float, float, float]:
    """Encodes coordinate relative to base.

    This function takes coordinates of 2 points and encodes
    second point relative to the first. The logic behind the
    encoding is to get distance and bearing, which are like
    polar coordinates, but are in fact geodesic, then convert
    angular bearing into its cosine and sine to cyclically
    (trigonometrically) encode the angle.

    Parameters
    ----------
    latitude1 : _type_
        base latitude
    longitude1 : _type_
        base longitude
    latitude2 : _type_
        encoded point's latitude
    longitude2 : _type_
        encoded point's longitude

    Returns
    -------
    tuple[float, float, float]
        tuple[distance, bearing_sin, bearing_cos]
    """
    point1 = (latitude1, longitude1)
    point2 = (latitude2, longitude2)

    # Geodesic distance in km
    distance = geodesic(point1, point2).km

    # Calculate initial bearing in radians
    lat1_rad = np.radians(latitude1)
    lon1_rad = np.radians(longitude1)
    lat2_rad = np.radians(latitude2)
    lon2_rad = np.radians(longitude2)

    delta_lon = lon2_rad - lon1_rad

    y = np.sin(delta_lon) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(
        lat2_rad
    ) * np.cos(delta_lon)

    initial_bearing_radians = np.arctan2(y, x)

    # Apply cyclic transformation
    bearing_sin = np.sin(initial_bearing_radians)
    bearing_cos = np.cos(initial_bearing_radians)

    return distance, bearing_sin, bearing_cos


def split_neighbourhood_id(
    neighborhood_id: str,
    locator: Nominatim | None = None,
    center_point: Location | None = None,
) -> tuple[str, str, float | None, float, float, float, float, float]:
    """Splits a neighborhood_id into separate features.

    neighborhood_id has the following format:
    "Neighborhood {int}: {neighborhood name} ({price per square meter or 'None'} €/m2) - District {int}: {district name}"
    This allows separation into a tuple of (neighborhood_name, district_name, sq_mt_price).

    Next step is deriving and encoding coordinates from the location. The encoding is polar coordinates with
    geodesic distance from the city's center and cyclically (trigonometrically) encoded bearing to the point.
    The distance, sin and cos are return for neighborhood.

    Parameters
    ----------
    neighborhood_id : str
        A string of the following format:
        "Neighborhood {int}: {neighborhood_name} ({sq_mt_price} €/m2) - District {int}: {district_name}"

    Returns
    -------
    tuple[str, str, float | None, float, float, float]
        (neighborhood_name, district_name, sq_mt_price, center_distance_neighborhood, bearing_sin_neighborhood,
        bearing_cos_neighborhood, neighborhood_latitude, neighborhood_longitude)
    """

    if locator is None:
        locator = Nominatim(user_agent="my_geocoding_app")

    if center_point is None:
        center_point = locator.geocode("Madrid")

    neighborhood_name = neighborhood_id.split(" (")[0].split(": ")[1]
    neighborhood_name = neighborhood_name.split("-")[0].strip().replace(" de ", " ")
    sq_mt_price = neighborhood_id.split("(")[1].split(" €/m2)")[0]

    # Since isdigit and isnumeric don't capture floats, a try clause is needed
    try:
        sq_mt_price = float(sq_mt_price)
    except ValueError:
        sq_mt_price = None
    district_name = neighborhood_id.split(": ")[-1]

    # Get distrit coordinates
    district_query = f"Madrid, {district_name}"
    try:
        district_location = _geocode_retry(locator, district_query)
        district_latitude, district_longitude = (
            district_location.latitude,
            district_location.longitude,
        )
    except GeocoderTimedOut as e:
        logger.exception(f"Query '{district_query}' Failed.")
        raise e

    # Get neighborhood coordinates
    neighborhood_query = f"Madrid, {neighborhood_name}"
    try:
        neighborhood_location = _geocode_retry(locator, neighborhood_query)
        if neighborhood_location is None:
            neighborhood_latitude, neighborhood_longitude = (
                district_latitude,
                district_longitude,
            )
        else:
            neighborhood_latitude, neighborhood_longitude = (
                neighborhood_location.latitude,
                neighborhood_location.longitude,
            )
    except GeocoderTimedOut as e:
        logger.exception(f"Query '{neighborhood_query}' Failed.")
        raise e

    # Convert coordinates into distance and bearing from city's center,
    # better alternative to the polar coordinates
    center_distance_neighborhood, bearing_sin_neighborhood, bearing_cos_neighborhood = (
        get_distance_and_bearing(
            center_point.latitude,
            center_point.longitude,
            neighborhood_latitude,
            neighborhood_longitude,
        )
    )

    return (
        neighborhood_name,
        district_name,
        sq_mt_price,
        center_distance_neighborhood,
        bearing_sin_neighborhood,
        bearing_cos_neighborhood,
        neighborhood_latitude,
        neighborhood_longitude,
    )


@FunctionTransformer
def process_neighborhoods(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepares and encodes neihborhoods' features.

    It splits neighborhood ids, then using geopy derives coordinats
    for the districts and neighborhoods. Then it encodes the coordinates
    into geodesic polar ones - bearing transformation with geodesic
    distance and trigonometric cyclic encoding. Fills out gaps in price per
    square meter data and handles geopy errors in neighborhood location
    via using districts' data.

    Parameters
    ----------
    frame : pd.DataFrame
        previous step frame

    Returns
    -------
    pd.DataFrame
        frame with added features

    Raises
    ------
    e
        Errors impossible to handle. These arise if geopy completely
        fails due to network errors or poor data.
    """
    # Read or create a table for all neighborhood ids
    neighborhoods_data_path = root_path / "data" / "neighborhoods.csv"
    if (neighborhoods_data_path).exists():
        neighborhoods_data = pd.read_csv(neighborhoods_data_path, index_col=0)
    else:
        neighborhood_id_mapping = {}
        locator = Nominatim(user_agent="my_geocoding_app", timeout=LOCATOR_TIMEOUT)
        center_point = locator.geocode("Madrid")
        for idx, id in tqdm(list(enumerate(frame["neighborhood_id"].unique()))):
            if id not in neighborhood_id_mapping:
                try:
                    neighborhood_id_mapping[id] = pd.Series(
                        split_neighbourhood_id(
                            id, center_point=center_point, locator=locator
                        ),
                        index=NEIGHBORHOOD_FEATURE_NAMES,
                    )
                except Exception as e:
                    logger.exception(f"{id}, index: {idx}")
                    raise e
        neighborhoods_data = pd.DataFrame(neighborhood_id_mapping).T
        neighborhoods_data.to_csv(neighborhoods_data_path)

    # The sq_mt_price restoration part
    ids_to_restore = list(
        neighborhoods_data[neighborhoods_data["sq_mt_price"].isna()].index
    )

    distances = {
        restore_id: {
            target_id: geodesic(
                (
                    neighborhoods_data.loc[restore_id]["latitude"],
                    neighborhoods_data.loc[restore_id]["longitude"],
                ),
                (
                    neighborhoods_data.loc[target_id]["latitude"],
                    neighborhoods_data.loc[target_id]["longitude"],
                ),
            ).km
            for target_id in neighborhoods_data[
                ~neighborhoods_data.index.isin(ids_to_restore)
            ].index
        }
        for restore_id in ids_to_restore
    }
    distances_df = pd.DataFrame(distances).T
    nearest_neighbors = distances_df.apply(
        lambda candidates: pd.Series(
            candidates.nsmallest(RESTORE_PRICE_NEIGHBORS).index
        ),
        axis=1,
    )
    restored_prices = nearest_neighbors.apply(
        lambda neighbors: np.mean(
            [neighborhoods_data.loc[neighbor, "sq_mt_price"] for neighbor in neighbors]
        ),
        axis=1,
    )
    neighborhoods_data["sq_mt_price"] = neighborhoods_data["sq_mt_price"].fillna(
        restored_prices
    )

    # Once prices are restored, the features should be mapped onto the dataset
    return frame.drop(columns=["latitude", "longitude"]).merge(
        neighborhoods_data, how="left", left_on="neighborhood_id", right_index=True
    )


@FunctionTransformer
def process_boolean_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepares boolean features.

    Fills incmplete and polar features with False. For orientation features adds
      new boolean feature of the orientation's presence.

    Parameters
    ----------
    frame : pd.DataFrame
        previous step frame

    Returns
    -------
    pd.DataFrame
        frame with added and processed features
    """
    # handle polar features
    frame[POLAR_BOOLEAN_FEATURES] = frame[POLAR_BOOLEAN_FEATURES].fillna(False)

    # handle incomplete features
    frame[INCOMPLETE_BOOLEAN_FEATURES] = frame[INCOMPLETE_BOOLEAN_FEATURES].fillna(
        False
    )
    frame["is_orientation_stated"] = frame[INCOMPLETE_BOOLEAN_FEATURES].any(axis=1)

    # Convert to binary
    frame[NON_EMPTY_FEATURES] = frame[NON_EMPTY_FEATURES].astype(int)
    return frame


@FunctionTransformer
def process_ordinal_features(frame: pd.DataFrame) -> pd.DataFrame:
    # Process energy certificates. Set as indices in ordered list
    frame["energy_certificate_provided"] = frame["energy_certificate"] == "no indicado"
    frame["energy_certificate"] = frame["energy_certificate"].apply(
        lambda certificate: ENERGY_CERTIFICATE_ORDER.index(certificate)
    )

    # Process built_year
    frame["built_year"][~frame["built_year"].between(0, datetime.now().year)] = None
    labels = np.linspace(0, 1, 21).round(3)
    built_year_notna_mask = ~frame["built_year"].isna()
    frame["built_year"][built_year_notna_mask] = pd.qcut(
        frame["built_year"][built_year_notna_mask], q=21, labels=labels
    ).astype(float)
    frame["built_year"][frame["built_year"].isna()] = 0

    # Process n_rooms. Minmax transform
    frame["n_rooms"] = (frame["n_rooms"] - frame["n_rooms"].min()) / (
        frame["n_rooms"].max() - frame["n_rooms"].min()
    )

    # Process n_bathrooms. Restore from known samples
    bathrooms_stated_mask = ~frame["n_bathrooms"].isna()
    bathroom_coefficient = (
        frame[bathrooms_stated_mask]["n_rooms"]
        / frame[bathrooms_stated_mask]["n_bathrooms"]
    ).mean()
    frame["n_bathrooms"][~bathrooms_stated_mask] = (
        (frame[~bathrooms_stated_mask]["n_rooms"] / bathroom_coefficient)
        .round()
        .astype(int)
    )

    # Process house_type_id. Set as indices in ordered list
    frame["house_type_id"] = frame["house_type_id"].apply(
        lambda house_type_id: HOUSE_TYPE_ORDER.index(house_type_id)
    )

    return frame
