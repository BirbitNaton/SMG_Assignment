from pathlib import Path

import numpy as np

DATA_PATH = Path(__file__).parent
DEFAULT_DATA_PATH = Path(f"{DATA_PATH}/houses_Madrid.csv")
DEFAULT_SCHEMA_PATH = Path(f"{DATA_PATH}/dataset_schema.json")

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
