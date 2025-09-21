import json
import warnings
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
from sklearn.pipeline import FunctionTransformer
from tenacity import retry, stop_after_attempt, wait_fixed
from tqdm.auto import tqdm

from task1.data.utils import (
    DATA_PATH,
    ENERGY_CERTIFICATE_ORDER,
    HOUSE_TYPE_ORDER,
    INCOMPLETE_BOOLEAN_FEATURES,
    LOCATOR_TIMEOUT,
    NEIGHBORHOOD_FEATURE_NAMES,
    NON_EMPTY_FEATURES,
    POLAR_BOOLEAN_FEATURES,
    RESTORE_PRICE_NEIGHBORS,
    RETRY_WAIT,
    STOP_AFTER_ATTEMPT,
)

warnings.filterwarnings("ignore")

root_path = Path(__file__).parent.parent
dataset_filename = "houses_Madrid.csv"
save_path = DATA_PATH / dataset_filename
logger = getLogger()
dataset_schema = json.load((DATA_PATH / "dataset_schema.json").open())


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


def convert_target(target: pd.Series) -> pd.Series:
    """Applies target transformations.

    Applies log transform to the target, since it was
    shown to exhibit lognormal distribution earlier.

    Parameters
    ----------
    frame : pd.Series
        initial target column

    Returns
    -------
    pd.Series
        logtransformed target
    """

    return np.log(target)


def revert_target(target: pd.Series) -> pd.Series:
    """Reverts target transformations.

    Applies exponential function to revert log transform to the target.

    Parameters
    ----------
    frame : pd.Series
        previously logtransformed target column or a prediction array

    Returns
    -------
    pd.DataFrame
        target converted back to original scale
    """

    return np.exp(target)


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

    # frame = frame[
    #     (sq_mt_built_present & sq_mt_built_meaningful)
    #     | (sq_mt_useful_present & sq_mt_useful_meaningful)
    # ]

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
    frame["sq_mt_built_present"] = (
        sq_mt_built_present & sq_mt_built_meaningful
    ).astype(int)
    frame["sq_mt_useful_present"] = (
        sq_mt_useful_present & sq_mt_useful_meaningful
    ).astype(int)

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
    return frame.drop(columns=["latitude", "longitude"], errors="ignore").merge(
        neighborhoods_data, how="left", left_on="neighborhood_id", right_index=True
    )


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


def process_ordinal_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepares ordinal features.

    Processes certain types of ordinal features separately.
    * energy_certificate - establishes order and adds presence feature
    * built_year - handles failed values and bins everything else,
        sets Nones to 0
    * n_bathrooms - restored via n_rooms
    * house_type_id - establishes order, None first


    Parameters
    ----------
    frame : pd.DataFrame
        previous step frame

    Returns
    -------
    pd.DataFrame
        frame with added and processed ordinal features
    """
    # Process energy certificates. Set as indices in ordered list
    frame["energy_certificate"] = frame["energy_certificate"].fillna("no indicado")
    frame["energy_certificate_provided"] = (
        frame["energy_certificate"] == "no indicado"
    ).astype(int)
    frame["energy_certificate"] = frame["energy_certificate"].apply(
        lambda certificate: ENERGY_CERTIFICATE_ORDER.index(certificate)
    )

    # Process built_year
    frame["built_year"][~frame["built_year"].between(0, datetime.now().year)] = None
    built_year_na_mask = frame["built_year"].isna()
    bins = [0, 1800, 1900, 1950, 1970, 1990, *list(range(2010, 2028, 2))]
    labels = [1800, 1900, 1950, 1970, 1990, *list(range(2010, 2028, 2))]
    frame.loc[~built_year_na_mask, "built_year"] = pd.cut(
        frame.loc[~built_year_na_mask, "built_year"],
        bins=bins,
        labels=labels,
    ).astype(float)
    frame["built_year"][built_year_na_mask] = 0

    # Process n_bathrooms. Restore from known samples
    bathrooms_stated_mask = ~frame["n_bathrooms"].isna()
    bathroom_coefficient = (
        frame[bathrooms_stated_mask]["n_rooms"]
        / frame[bathrooms_stated_mask]["n_bathrooms"]
    ).mean()
    frame["n_bathrooms"][~bathrooms_stated_mask] = (
        frame[~bathrooms_stated_mask]["n_rooms"] / bathroom_coefficient
    )

    # Process house_type_id. Set as indices in ordered list
    house_type_order_mapping = dict(tuple(enumerate(HOUSE_TYPE_ORDER)))
    frame["house_type_id"] = frame["house_type_id"].map(house_type_order_mapping)

    return frame


def feature_selector(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only required features.

    Uses preloaded dataset_schema to select features for the model training.

    Parameters
    ----------
    frame : pd.DataFrame
        previous step frame with prepared features

    Returns
    -------
    pd.DataFrame
        frame with only the selected features
    """
    return frame[dataset_schema["features"]]


def get_preprocessing_stages():
    """Get the preprocessing stages for the pipeline.

    Returns
    -------
    list
        A list of tuples representing the preprocessing stages.
    """
    return [
        ("process_square_meters", FunctionTransformer(process_square_meters)),
        ("process_neighborhoods", FunctionTransformer(process_neighborhoods)),
        ("process_boolean_features", FunctionTransformer(process_boolean_features)),
        ("process_ordinal_features", FunctionTransformer(process_ordinal_features)),
        ("feature_selector", FunctionTransformer(feature_selector)),
    ]
