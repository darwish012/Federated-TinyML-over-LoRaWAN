"""Restricted fresh roster for V41R1; raw parsing remains V41-identical."""
from raw_online_federated_v41_data import *  # noqa: F401,F403
from raw_online_federated_v41_data import TargetDevice


def target_devices() -> list[TargetDevice]:
    result = [TargetDevice(test, bearing) for test in (2, 5) for bearing in (1, 2)]
    if len(result) != 4 or len({item.client_id for item in result}) != 4:
        raise AssertionError("V41R1 needs four unique physical-bearing clients")
    return result

