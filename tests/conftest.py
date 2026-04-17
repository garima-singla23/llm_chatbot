import os

import pytest


@pytest.fixture(scope="session")
def flightapi_client():
    from apis.flightapi_client import FlightAPIClient

    return FlightAPIClient(api_key=os.getenv("FLIGHTAPI_KEY"))
