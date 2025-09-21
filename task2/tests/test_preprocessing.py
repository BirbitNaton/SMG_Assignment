import numpy as np
import pandas as pd

from task1.data.preprocessing import (
    convert_target,
    process_boolean_features,
    process_neighborhoods,
    process_ordinal_features,
    process_square_meters,
    revert_target,
    split_neighbourhood_id,
)
from task1.data.utils import DEFAULT_DATA_PATH, NON_EMPTY_FEATURES

ORIGINAL_FRAME = pd.read_csv(DEFAULT_DATA_PATH).iloc[:20]


def test_target_transform():
    """Tests target transformation.

    Ensures that target transform is reversible via a corresponding function.
    """
    # Arrange
    input_target = pd.Series(list(range(20)))

    # Act
    target_transformed = convert_target(input_target)
    target_reverted = revert_target(target_transformed)

    # Assert
    assert np.isclose(input_target, target_reverted).all()


def test_process_sq_mt():
    """Tests square meter features' transformation.

    Compares transform results on a slice of the dataset with expected values.
    """
    # Arrange
    compare_columns = [
        "sq_mt_built",
        "sq_mt_useful",
        "n_rooms",
        "n_bathrooms",
        "sq_mt_built_present",
        "sq_mt_useful_present",
        "sq_mt_built_proc",
        "sq_mt_useful_proc",
    ]
    reference_dataframe = pd.DataFrame(
        {
            "sq_mt_built": {0: 64.0, 1: 70.0, 2: 94.0, 3: 64.0, 4: 108.0},
            "sq_mt_useful": {0: 60.0, 1: np.nan, 2: 54.0, 3: np.nan, 4: 90.0},
            "n_rooms": {0: 2, 1: 3, 2: 2, 3: 2, 4: 2},
            "n_bathrooms": {0: 1.0, 1: 1.0, 2: 2.0, 3: 1.0, 4: 2.0},
            "sq_mt_built_present": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
            "sq_mt_useful_present": {0: 1, 1: 0, 2: 1, 3: 0, 4: 1},
            "sq_mt_built_proc": {
                0: 4.1588830833596715,
                1: 4.248495242049359,
                2: 4.543294782270004,
                3: 4.1588830833596715,
                4: 4.68213122712422,
            },
            "sq_mt_useful_proc": {
                0: 4.0943445622221,
                1: 4.0661736852554045,
                2: 3.9889840465642745,
                3: 3.9765615265657175,
                4: 4.499809670330265,
            },
        }
    )

    # Act
    square_meters_processed = process_square_meters(ORIGINAL_FRAME.iloc[:5]).loc[
        :, compare_columns
    ]

    # Assert
    assert square_meters_processed.equals(reference_dataframe)


def test_process_neighborhoods():
    """Tests neighborhoods features' transformation.

    Compares transform results on a slice of the dataset with expected values.
    """
    # Arrange
    reference_columns = [
        "center_distance",
        "bearing_sin",
        "bearing_cos",
        "latitude",
        "longitude",
    ]
    reference_dataframe = pd.DataFrame(
        {
            "center_distance": {
                0: 8.383018128565531,
                1: 6.7245001856268285,
                2: 7.92724909671289,
                3: 7.92724909671289,
                4: 6.71619861975953,
            },
            "bearing_sin": {
                0: 0.1026656766854674,
                1: 0.0496260397317187,
                2: -0.0788674720355628,
                3: -0.0788674720355628,
                4: 0.1780958815492608,
            },
            "bearing_cos": {
                0: -0.9947159186575408,
                1: -0.9987678690168932,
                2: -0.9968851096564336,
                3: -0.9968851096564336,
                4: -0.9840131386191912,
            },
            "latitude": {
                0: 40.3416121,
                1: 40.3562217,
                2: 40.3455389,
                3: 40.3455389,
                4: 40.3571948,
            },
            "longitude": {
                0: -3.6934141,
                1: -3.6996388,
                2: -3.7109697,
                3: -3.7109697,
                4: -3.6894484,
            },
        }
    )

    # Act
    neighborhoods_processed = process_neighborhoods(ORIGINAL_FRAME.iloc[:5]).loc[
        :, reference_columns
    ]

    # Assert
    assert neighborhoods_processed.equals(reference_dataframe)


def test_split_neighbourhood_id():
    """Tests correctness of the neighbourhood_id slicing.

    Derives coordinates from the string, processes them and compares to expected value.
    """
    # Arrange
    id = "Neighborhood 135: San Cristóbal (1308.89 €/m2) - District 21: Villaverde"

    # Act
    split_id = split_neighbourhood_id(id)

    # Assert
    assert split_id == (
        "San Cristóbal",
        "Villaverde",
        1308.89,
        8.383018128565531,
        np.float64(0.10266567668546742),
        np.float64(-0.9947159186575407),
        40.3416121,
        -3.6934141,
    )


def test_process_boolean_and_ordinal_features():
    """Tests booleans' and ordinals' handling.

    Prepares boolean and ordinal features, and validates their correctness.
    """
    # Arrange
    reference_dataframe = pd.DataFrame(
        {
            "is_orientation_north": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
            "has_central_heating": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "is_orientation_stated": {0: 1, 1: 0, 2: 0, 3: 1, 4: 1},
            "is_renewal_needed": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0},
            "is_exterior": {0: 1, 1: 1, 2: 1, 3: 1, 4: 1},
            "is_orientation_south": {0: 0, 1: 0, 2: 0, 3: 1, 4: 1},
            "has_fitted_wardrobes": {0: 0, 1: 1, 2: 1, 3: 0, 4: 1},
            "has_balcony": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "is_accessible": {0: 0, 1: 0, 2: 0, 3: 1, 4: 0},
            "is_floor_under": {0: 0, 1: 0, 2: 0, 3: 1, 4: 0},
            "has_ac": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1},
            "is_new_development": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "has_garden": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0},
            "has_lift": {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
            "has_terrace": {0: 0, 1: 1, 2: 0, 3: 0, 4: 0},
            "has_individual_heating": {0: 0, 1: 0, 2: 1, 3: 0, 4: 0},
            "has_parking": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
            "is_orientation_west": {0: 1, 1: 0, 2: 0, 3: 0, 4: 1},
            "is_orientation_east": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
            "has_green_zones": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
            "has_storage_room": {0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
            "is_parking_included_in_price": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
            "has_pool": {0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
        }
    )

    # Act
    processed_boolean_features = (
        process_ordinal_features(process_boolean_features(ORIGINAL_FRAME))
        .loc[:, NON_EMPTY_FEATURES]
        .iloc[:5]
    )

    # Assert
    assert processed_boolean_features.sort_index(axis=1).equals(
        reference_dataframe.sort_index(axis=1)
    )
