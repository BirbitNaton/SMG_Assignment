from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from task1.api import main
from task1.api.main import app

DEFAULT_PAYLOAD_SINGLE = {
    "sq_mt_built_proc": 0,
    "sq_mt_built_present": 0,
    "sq_mt_useful_present": 0,
    "sq_mt_price": 0,
    "center_distance": 0,
    "bearing_sin": 0,
    "bearing_cos": 0,
    "latitude": 0,
    "longitude": 0,
    "has_balcony": 0,
    "has_ac": 0,
    "has_terrace": 0,
    "has_pool": 0,
    "is_exterior": 0,
    "is_renewal_needed": 0,
    "is_orientation_north": 0,
    "is_parking_included_in_price": 0,
    "is_orientation_stated": 0,
    "has_fitted_wardrobes": 0,
    "has_parking": 0,
    "is_orientation_west": 0,
    "has_central_heating": 0,
    "has_green_zones": 0,
    "is_accessible": 0,
    "is_new_development": 0,
    "is_orientation_east": 0,
    "is_orientation_south": 0,
    "has_lift": 0,
    "has_storage_room": 0,
    "has_garden": 0,
    "has_individual_heating": 0,
    "is_floor_under": 0,
    "energy_certificate_provided": 0,
    "energy_certificate": 0,
    "built_year": 0,
    "n_rooms": 0,
    "n_bathrooms": 0,
    "house_type_id": 0,
}


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    dummy_model = MagicMock()
    dummy_model.predict.return_value = [1, 2, 3]
    monkeypatch.setattr(main, "model", dummy_model)


@pytest.fixture(autouse=True)
def mock_model_info(monkeypatch):
    model_info = MagicMock()
    model_info.name = None
    monkeypatch.setattr(main, "model_info", model_info)


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_model_info():
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as client:
        response = await client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "latest_version" in data


@pytest.mark.asyncio
async def test_predict():
    payload = DEFAULT_PAYLOAD_SINGLE
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as client:
        response = await client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()


@pytest.mark.asyncio
async def test_batch_predict():
    payload = {"inputs": [DEFAULT_PAYLOAD_SINGLE]}
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://127.0.0.1:8000"
    ) as client:
        response = await client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
