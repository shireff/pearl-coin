import torch
from miner_base.commitment_hash import CommitmentHasher
from miner_base.gpu_matmul_config import GPUMatmulConfigFactory
from miner_utils import get_logger
from pearl_gemm import (
    commitment_hash_from_merkle_roots,
    gemm,
    get_host_signal_sync_size,
    get_required_scratchpad_bytes,
    make_pow_target_tensor,
    noise_gen,
    noisy_gemm,
    run_tensor_hash,
)

from .callbacks import (
    StatusCheckCallback,
)
from .config import config
from .mining_state import (
    get_async_manager,
    get_pinned_pool,
)

_LOGGER = get_logger("vllm.pearl_miner")

_PERSISTENT_BUFFERS: dict[tuple, torch.Tensor] = {}

_HOST_SIGNAL_SYNC_SIZE = get_host_signal_sync_size()


def _get_persistent_buffer(
    name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    key = (name, shape, dtype, device.index)
    buf = _PERSISTENT_BUFFERS.get(key)
    if buf is None or buf.shape != shape or buf.dtype != dtype or buf.device != device:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _PERSISTENT_BUFFERS[key] = buf
    return buf


def _clear_persistent_buffers() -> None:
    _PERSISTENT_BUFFERS.clear()


def _acquire_pinned_buffer() -> torch.Tensor:
    return get_pinned_pool().acquire()


def _release_pinned_buffer(buffer: torch.Tensor) -> None:
    get_pinned_pool().release(buffer)


def pearl_gemm_vanilla(
    A: torch.Tensor,
    B: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Performs standard quantized matrix multiplication without mining operations.

    Computes C = A @ B.T using optimized CUDA kernels for int8 quantized inputs.

    :param A: Input matrix A (int8, quantized)
    :param B: Input matrix B (int8, quantized)
    :param scale_a: Quantization scale factors for matrix A
    :param scale_b: Quantization scale factors for matrix B
    :param out_dtype: Output data type (bfloat16 or float16)
    :return: Result matrix C
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    A_scales = scale_a
    B_scales = scale_b

    C = torch.empty((A.shape[0], B.shape[0]), dtype=out_dtype, device=A.device)

    gemm(
        A=A,
        B=B,
        A_scales=A_scales,
        B_scales=B_scales,
        C=C,
        tile_size_m=config.settings.tile_size_m,
        tile_size_n=config.settings.tile_size_n,
        tile_size_k=config.settings.tile_size_k,
    )

    return C


def _run_tensor_hash_on_stream(
    tensor: torch.Tensor,
    key_tensor: torch.Tensor,
    output_hash: torch.Tensor,
    scratchpad: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    with torch.cuda.stream(stream):
        run_tensor_hash(
            tensor.to(torch.uint8),
            key_tensor,
            output_hash,
            scratchpad,
        )


def _generate_noise_factors_streamed(
    m: int,
    n: int,
    k: int,
    r: int,
    commitment_hash_A: torch.Tensor,
    commitment_hash_B: torch.Tensor,
    device: torch.device,
    stream: torch.cuda.Stream,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Generates cryptographic noise factors on a dedicated CUDA stream.

    Runs noise_gen asynchronously on the provided stream so it overlaps
    with other GPU work, improving GPU utilization.
    """
    EAL = _get_persistent_buffer("EAL", (m, r), torch.int8, device)
    EBR = _get_persistent_buffer("EBR", (n, r), torch.int8, device)
    EAR_R_major = _get_persistent_buffer("EAR_R_major", (k, r), torch.int8, device)
    EBL_R_major = _get_persistent_buffer("EBL_R_major", (k, r), torch.int8, device)
    EAR_K_major = _get_persistent_buffer("EAR_K_major", (r, k), torch.int8, device)
    EBL_K_major = _get_persistent_buffer("EBL_K_major", (r, k), torch.int8, device)
    EAL_fp16 = _get_persistent_buffer("EAL_fp16", (m, r), torch.float16, device)
    EBR_fp16 = _get_persistent_buffer("EBR_fp16", (n, r), torch.float16, device)

    with torch.cuda.stream(stream):
        noise_gen(
            R=r,
            EAL=EAL,
            EAL_fp16=EAL_fp16,
            EAR_R_major=EAR_R_major,
            EAR_K_major=EAR_K_major,
            EBL_R_major=EBL_R_major,
            EBL_K_major=EBL_K_major,
            EBR=EBR,
            EBR_fp16=EBR_fp16,
            key_A=commitment_hash_A,
            key_B=commitment_hash_B,
        )

    return (
        EAL,
        EAR_R_major,
        EBL_R_major,
        EAR_K_major,
        EBL_K_major,
        EBR,
        EAL_fp16,
        EBR_fp16,
    )


def pearl_gemm_noisy(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    layer: torch.nn.Module | None = None,
    submit_block: bool = True,
) -> torch.Tensor:
    """
    Performs quantized matrix multiplication with cryptographic noise for blockchain mining.

    Computes C = A @ B.T while generating proof-of-work hashes from intermediate computations
    using optimized CUDA kernels for noise generation and matrix operations.

    :param a: Input matrix A (int8, quantized)
    :param b: Input matrix B (int8, quantized)
    :param scale_a: Quantization scale factors for matrix A
    :param scale_b: Quantization scale factors for matrix B
    :param out_dtype: Output data type (bfloat16 or float16)
    :param layer: Layer containing weight tensors
    :param submit_block: Whether to submit mining results
    :return: Result matrix C
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16

    m = a.shape[0]
    n = b.shape[0]
    k = a.shape[1]
    r = config.settings.noise_rank
    device = a.device

    A = a
    B = b
    A_scales = scale_a
    B_scales = scale_b

    C = _get_persistent_buffer("C", (m, n), out_dtype, device)
    scratchpad = _get_persistent_buffer(
        "scratchpad",
        (get_required_scratchpad_bytes(max(m * k, n * k)),),
        torch.uint8,
        device,
    )
    key_tensor = _get_persistent_buffer("key_tensor", (32,), torch.uint8, device)
    A_tensor_hash = _get_persistent_buffer("A_tensor_hash", (32,), torch.uint8, device)
    B_tensor_hash = _get_persistent_buffer("B_tensor_hash", (32,), torch.uint8, device)
    commitment_hash_A = _get_persistent_buffer("commitment_hash_A", (32,), torch.uint8, device)
    commitment_hash_B = _get_persistent_buffer("commitment_hash_B", (32,), torch.uint8, device)
    pow_target_tensor = _get_persistent_buffer("pow_target", (8,), torch.uint32, device)
    host_signal_sync = _get_persistent_buffer(
        "host_signal_sync", (_HOST_SIGNAL_SYNC_SIZE,), torch.int8, device
    )

    EAL = _get_persistent_buffer("EAL", (m, r), torch.int8, device)
    EBR = _get_persistent_buffer("EBR", (n, r), torch.int8, device)
    EAR_R_major = _get_persistent_buffer("EAR_R_major", (k, r), torch.int8, device)
    EBL_R_major = _get_persistent_buffer("EBL_R_major", (k, r), torch.int8, device)
    EAR_K_major = _get_persistent_buffer("EAR_K_major", (r, k), torch.int8, device)
    EBL_K_major = _get_persistent_buffer("EBL_K_major", (r, k), torch.int8, device)
    EAL_fp16 = _get_persistent_buffer("EAL_fp16", (m, r), torch.float16, device)
    EBR_fp16 = _get_persistent_buffer("EBR_fp16", (n, r), torch.float16, device)

    BpEB = _get_persistent_buffer("BpEB", (n, k), torch.int8, device)
    EARxBpEB = _get_persistent_buffer("EARxBpEB", (n, r), torch.float16, device)
    ApEA = _get_persistent_buffer("ApEA", (m, k), torch.int8, device)
    A_E_BL = _get_persistent_buffer("A_E_BL", (m, r), torch.float16, device)

    matmul_config = GPUMatmulConfigFactory.create(k=k, noise_rank=r)

    mining_job = get_async_manager().get_mining_job()
    adjusted_target = mining_job.adjust_target(mining_config=matmul_config.mining_config)

    hash_key = CommitmentHasher.get_key(
        mining_job.incomplete_header_bytes, matmul_config.mining_config
    )

    key_tensor.copy_(torch.frombuffer(bytearray(hash_key), dtype=torch.uint8))

    stream_a = torch.cuda.Stream(device=device)
    stream_b = torch.cuda.Stream(device=device)

    _run_tensor_hash_on_stream(A.to(torch.uint8), key_tensor, A_tensor_hash, scratchpad, stream_a)
    _run_tensor_hash_on_stream(B.to(torch.uint8), key_tensor, B_tensor_hash, scratchpad, stream_b)

    torch.cuda.current_stream().wait_stream(stream_a)
    torch.cuda.current_stream().wait_stream(stream_b)

    commitment_hash_from_merkle_roots(
        A_tensor_hash, B_tensor_hash, key_tensor,
        commitment_hash_A, commitment_hash_B,
    )

    noise_stream = torch.cuda.Stream(device=device)
    noise_factors = _generate_noise_factors_streamed(
        m, n, k, r, commitment_hash_A, commitment_hash_B, device, noise_stream
    )

    torch.cuda.current_stream().wait_stream(noise_stream)

    (
        EAL, EAR_R_major, EBL_R_major, EAR_K_major, EBL_K_major, EBR, EAL_fp16, EBR_fp16
    ) = noise_factors

    pow_target_tensor.copy_(make_pow_target_tensor(adjusted_target))

    host_signal_header_pinned = _acquire_pinned_buffer()

    noisy_gemm(
        A=A,
        B=B,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        EAR_R_major=EAR_R_major,
        EBL_R_major=EBL_R_major,
        EAR_K_major=EAR_K_major,
        EBL_K_major=EBL_K_major,
        AxEBL_fp16=A_E_BL,
        EARxBpEB_fp16=EARxBpEB,
        ApEA=ApEA,
        BpEB=BpEB,
        A_scales=A_scales,
        B_scales=B_scales,
        C=C,
        host_signal_header_pinned=host_signal_header_pinned,
        host_signal_sync=host_signal_sync,
        pow_target=pow_target_tensor,
        pow_key=commitment_hash_A.view(torch.uint32),
        tile_size_m=config.settings.tile_size_m,
        tile_size_n=config.settings.tile_size_n,
        tile_size_k=config.settings.tile_size_k,
        pipeline_stages=4,
        cluster_size_m=2,
        cluster_size_n=2,
        run_noising_A=True,
        run_noising_B=True,
        skip_reduction=False,
        skip_denoising=False,
    )

    if submit_block:
        cuda_event = torch.cuda.Event()
        cuda_event.record()

        callback = StatusCheckCallback(
            host_signal_header_pinned=host_signal_header_pinned,
            commitment_hash_A_tensor=commitment_hash_A,
            commitment_hash_B_tensor=commitment_hash_B,
            A=A,
            B=B,
            mining_job=mining_job,
        )

        get_async_manager().schedule_status_check(cuda_event, callback)
    else:
        _release_pinned_buffer(host_signal_header_pinned)

    return C


def generate_noise_factors(
    m: int,
    n: int,
    k: int,
    r: int,
    commitment_hash_A: torch.Tensor,
    commitment_hash_B: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Generates cryptographic noise factors for noisy GEMM operations.

    Creates low-rank noise matrices using commitment hashes as seeds for deterministic
    noise generation in blockchain mining operations using optimized CUDA kernels.

    :param m: Number of rows in matrix A
    :param n: Number of rows in matrix B
    :param k: Number of columns in matrices A and B
    :param r: Rank of the noise matrices
    :param commitment_hash_A: Cryptographic hash for matrix A noise generation
    :param commitment_hash_B: Cryptographic hash for matrix B noise generation
    :param device: CUDA device for tensor allocation
    :return: Tuple of noise tensors (EAL, EAR_R_major, EBL_R_major,
             EAR_K_major, EBL_K_major, EBR, EAL_fp16, EBR_fp16)
    """
    EAL = torch.empty((m, r), dtype=torch.int8, device=device)
    EBR = torch.empty((n, r), dtype=torch.int8, device=device)

    EAR_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EBL_R_major = torch.empty((k, r), dtype=torch.int8, device=device)
    EAR_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    EBL_K_major = torch.empty((r, k), dtype=torch.int8, device=device)
    EAL_fp16 = torch.empty((m, r), dtype=torch.float16, device=device)
    EBR_fp16 = torch.empty((n, r), dtype=torch.float16, device=device)

    noise_gen(
        R=r,
        EAL=EAL,
        EAL_fp16=EAL_fp16,
        EAR_R_major=EAR_R_major,
        EAR_K_major=EAR_K_major,
        EBL_R_major=EBL_R_major,
        EBL_K_major=EBL_K_major,
        EBR=EBR,
        EBR_fp16=EBR_fp16,
        key_A=commitment_hash_A,
        key_B=commitment_hash_B,
    )
    return (
        EAL,
        EAR_R_major,
        EBL_R_major,
        EAR_K_major,
        EBL_K_major,
        EBR,
        EAL_fp16,
        EBR_fp16,
    )