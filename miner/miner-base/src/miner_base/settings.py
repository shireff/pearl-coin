from pydantic_settings import BaseSettings, SettingsConfigDict


class MinerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="miner_")

    noise_range: int = 128
    noise_rank: int = 256
    idxs_per_col: int = 4

    # GEMM tile sizes
    tile_size_m: int = 256
    tile_size_n: int = 512
    tile_size_k: int = 256

    # fmt: off
    # Hash tile pattern for the 256x512 tile
    rows_pattern: list[int] = [0, 8]
    cols_pattern: list[int] = [
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27,
    32, 33, 34, 35, 40, 41, 42, 43, 48, 49, 50, 51, 56, 57, 58, 59,
    64, 65, 66, 67, 72, 73, 74, 75, 80, 81, 82, 83, 88, 89, 90, 91,
    96, 97, 98, 99, 104, 105, 106, 107, 112, 113, 114, 115, 120, 121, 122, 123,
    128, 129, 130, 131, 136, 137, 138, 139, 144, 145, 146, 147, 152, 153, 154, 155,
    160, 161, 162, 163, 168, 169, 170, 171, 176, 177, 178, 179, 184, 185, 186, 187,
    192, 193, 194, 195, 200, 201, 202, 203, 208, 209, 210, 211, 216, 217, 218, 219,
    224, 225, 226, 227, 232, 233, 234, 235, 240, 241, 242, 243, 248, 249, 250, 251,
    ]
    # fmt: on

    pinned_pool_size: int = 256

    debug: bool = False
    print_header_hash: bool = False
    no_gateway: bool = False
    no_mining: bool = False
    skip_block_submission: bool = False
    no_vllm_plugin: bool = False

    enable_async_cuda_event_processing: bool = True
