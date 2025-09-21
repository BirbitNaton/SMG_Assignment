from typing import List

from pydantic import BaseModel


# Example input schema for a single prediction
class PredictionInput(BaseModel):
    """FastAPI input model.

    This list of features matches the one after preprocessing
    since some are generated there, some are restored, and some are dropped.

    If lost, refer to the models' artifacts in MLflow for the exact list.
    """

    sq_mt_built_proc: float
    sq_mt_built_present: float
    sq_mt_useful_present: float
    sq_mt_price: float
    center_distance: float
    bearing_sin: float
    bearing_cos: float
    latitude: float
    longitude: float
    has_balcony: float
    has_ac: float
    has_terrace: float
    has_pool: float
    is_exterior: float
    is_renewal_needed: float
    is_orientation_north: float
    is_parking_included_in_price: float
    is_orientation_stated: float
    has_fitted_wardrobes: float
    has_parking: float
    is_orientation_west: float
    has_central_heating: float
    has_green_zones: float
    is_accessible: float
    is_new_development: float
    is_orientation_east: float
    is_orientation_south: float
    has_lift: float
    has_storage_room: float
    has_garden: float
    has_individual_heating: float
    is_floor_under: float
    energy_certificate_provided: float
    energy_certificate: float
    built_year: float
    n_rooms: float
    n_bathrooms: float
    house_type_id: float | None


class BatchPredictionInput(BaseModel):
    """FastAPI input model for batch predictions.

    Used to validate and parse the input data for batch predictions.
    It contains a list of individual prediction inputs.
    """

    inputs: List[PredictionInput]
