from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratum_client import (
    build_full_nonce24_from_counter,
    build_full_nonce24_from_payload,
    build_submit_nonce_payload,
)


def test_nonce_layout_prefers_extranonce_prefix_and_22_byte_payload():
    extranonce = b"\x9b\xa2"
    counter = 0x123456

    nonce24 = build_full_nonce24_from_counter(extranonce, counter)
    assert len(nonce24) == 24
    assert nonce24[:2] == extranonce

    payload = build_submit_nonce_payload(nonce24)
    assert payload == nonce24[2:].hex()
    assert len(bytes.fromhex(payload)) == 22

    reconstructed = build_full_nonce24_from_payload(extranonce, bytes.fromhex(payload))
    assert reconstructed == nonce24
