"""
Test BLAKE3 keyed hash CUDA implementation against CPU reference.
"""
import pytest
import torch

from blake3 import blake3
from pearl_gemm.test_components import blake3_keyed_hash


class TestBlake3KeyedHash:
    """Test class for BLAKE3 keyed hash CUDA implementation."""

    @pytest.fixture
    def cpu_ref(self):
        """Return CPU reference function."""
        def ref(messages_np, key_np):
            out = []
            for i in range(len(messages_np)):
                h = blake3(messages_np[i].tobytes(), key=key_np.tobytes())
                out.append(list(h.digest()))
            return out
        return ref

    def test_keyed_hash_cuda_vs_cpu(self):
        """Test CUDA BLAKE3 keyed hash matches CPU blake3 crate."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(42)

        num_hashes = 16
        messages = torch.randint(0, 2**32, (num_hashes, 16), dtype=torch.uint32, device="cuda")
        key = torch.randint(0, 2**32, (8,), dtype=torch.uint32, device="cuda")

        cuda_output = blake3_keyed_hash(messages, key).cpu().numpy()

        messages_cpu = messages.cpu().numpy()
        key_cpu = key.cpu().numpy()
        expected = []
        for i in range(num_hashes):
            h = blake3(messages_cpu[i].tobytes(), key=key_cpu.tobytes())
            expected.append(list(h.digest()))
        expected = (torch.tensor(expected, dtype=torch.uint32).view(-1, 8).numpy())

        for i in range(num_hashes):
            np.testing.assert_array_equal(
                cuda_output[i],
                expected[i],
                err_msg=f"Hash mismatch for message {i}",
            )

    def test_keyed_hash_single_block(self):
        """Test single-block keyed hash matches CPU reference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        msg = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.uint32, device="cuda")
        key = torch.tensor([0] * 8, dtype=torch.uint32, device="cuda")

        cuda_output = blake3_keyed_hash(msg, key).cpu().numpy()

        expected = blake3(msg.cpu().numpy()[0].tobytes(), key=key.cpu().numpy().tobytes()).digest()
        expected_np = (torch.tensor(list(expected), dtype=torch.uint32).view(1, 8).numpy())

        np.testing.assert_array_equal(cuda_output[0], expected_np[0])

    def test_keyed_hash_all_zeros(self):
        """Test all-zero message and key."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        msg = torch.zeros((1, 16), dtype=torch.uint32, device="cuda")
        key = torch.zeros((8,), dtype=torch.uint32, device="cuda")

        cuda_output = blake3_keyed_hash(msg, key).cpu().numpy()

        expected = blake3(b'\x00' * 64, key=b'\x00' * 32).digest()
        expected_np = (torch.tensor(list(expected), dtype=torch.uint32).view(1, 8).numpy())

        np.testing.assert_array_equal(cuda_output[0], expected_np[0])

    def test_keyed_hash_different_keys(self):
        """Test that different keys produce different hashes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        msg = torch.randint(0, 2**32, (4, 16), dtype=torch.uint32, device="cuda")
        key1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device="cuda")
        key2 = torch.tensor([2, 0, 0, 0, 0, 0, 0, 0], dtype=torch.uint32, device="cuda")

        out1 = blake3_keyed_hash(msg, key1).cpu().numpy()
        out2 = blake3_keyed_hash(msg, key2).cpu().numpy()

        assert not np.array_equal(out1, out2), "Different keys should produce different hashes"

    def test_keyed_hash_different_messages(self):
        """Test that different messages produce different hashes."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        msg1 = torch.zeros((1, 16), dtype=torch.uint32, device="cuda")
        msg2 = torch.zeros((1, 16), dtype=torch.uint32, device="cuda")
        msg2[0, 0] = 1

        key = torch.randint(0, 2**32, (8,), dtype=torch.uint32, device="cuda")

        out1 = blake3_keyed_hash(msg1, key).cpu().numpy()
        out2 = blake3_keyed_hash(msg2, key).cpu().numpy()

        assert not np.array_equal(out1, out2), "Different messages should produce different hashes"

    def test_keyed_hash_deterministic(self):
        """Test that the function produces deterministic results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        torch.manual_seed(123)
        msg = torch.randint(0, 2**32, (8, 16), dtype=torch.uint32, device="cuda")
        key = torch.randint(0, 2**32, (8,), dtype=torch.uint32, device="cuda")

        out1 = blake3_keyed_hash(msg, key).cpu().numpy()
        out2 = blake3_keyed_hash(msg, key).cpu().numpy()

        np.testing.assert_array_equal(out1, out2, err_msg="Results are not deterministic")
