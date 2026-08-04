from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stratum_client import (
    build_full_nonce24_from_counter,
    build_full_nonce24_from_payload,
    build_submit_nonce_payload,
    select_stratum_target,
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


def test_target_prefers_pool_block_target_over_share_target():
    assert select_stratum_target(0x1234, 0x5678) == 0x1234
    assert select_stratum_target(0, 0x5678) == 0x5678
    assert select_stratum_target(0, 0) == 2**256 - 1
