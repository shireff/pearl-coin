# Kryptex Stratum Proxy for Pearl Node

This module provides a Stratum V1 mining proxy that bridges Kryptex Pool miners to the Pearl (PRL) blockchain node. It translates the Stratum mining protocol into Pearl node JSON-RPC calls, allowing Kryptex-compatible miners to mine Pearl through the local pearld node.

## Quick Start

```bash
cd proxy/kryptex
go run .
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PEARLD_RPC_URL` | `http://127.0.0.1:44107` | Pearl node JSON-RPC endpoint |
| `PEARLD_RPC_USER` | `rpcuser` | Pearl node RPC username |
| `PEARLD_RPC_PASSWORD` | `rpcpass` | Pearl node RPC password |
| `STRATUM_PORT` | `3333` | Stratum V1 listen port |
| `STRATUM_TLS` | `false` | Enable TLS for Stratum connections |
| `STRATUM_CERT_FILE` | | TLS certificate file path |
| `STRATUM_KEY_FILE` | | TLS key file path |
| `HEALTH_PORT` | `9900` | Health check HTTP port |
| `STRATUM_DIFFICULTY` | `1` | Default Stratum difficulty |
| `MAX_CONNECTIONS` | `1000` | Maximum concurrent miners |

## Supported Coins

| Coin | Pool Host | Port |
|------|-----------|------|
| PRL (Pearl) | `prl.kryptex.network` | `7048` |
| BTC (Bitcoin) | `btc.kryptex.network` | `7014` |
| LTC (Litecoin) | `ltc.kryptex.network` | `7016` |
| ETHW (EthereumPoW) | `ethw.kryptex.network` | `7034` |
| RVN (Ravencoin) | `rvn.kryptex.network` | `7031` |

## Connection Formats

Miners connect using the standard Kryptex wallet format:

```
mining_username.worker_name
```

Example with worker `MyRig1`:

```
krxabcdef12345.MyRig1
```

## Docker Compose

```yaml
services:
  kryptex-proxy:
    build:
      context: ../../
      dockerfile: proxy/Dockerfile.kryptex
    ports:
      - "3333:3333"    # Stratum V1
      - "9900:9900"    # Health check
    environment:
      PEARLD_RPC_URL: http://pearld:44107
      PEARLD_RPC_USER: rpcuser
      PEARLD_RPC_PASSWORD: rpcpass
      STRATUM_PORT: 3333
      STRATUM_DIFFICULTY: 1
    depends_on:
      - pearld
```

## Miner Configuration

### For Kryptex Miner (Windows)

1. Download Kryptex Miner from `https://kryptex.com`
2. Sign up and verify your email
3. In the pool configuration:
   - Pool address: `stratum+tcp://<proxy-host>:3333`
   - Wallet: your PRL mining username (e.g., `krxXXXXXX`)
   - Worker: your rig name (e.g., `MyRig1`)
   - Password: `x`

### For HiveOS

1. Create a wallet with type **Stratum**
2. Pool address: `stratum+tcp://<proxy-host>:3333`
3. Wallet: `<mining_username>.<worker_name>`
4. Password: `x`

### For RaveOS

1. Add a Stratum wallet
2. Pool: `<proxy-host>:3333`
3. Format: `<mining_username>.<worker_name>`

## Architecture

```
Miner (Stratum V1)
      |
      v
  Kryptex Proxy (3333)  <-- Stratum protocol
      |
      v  (JSON-RPC)
  Pearl Node / Proxy    <-- Pearl JSON-RPC API
```

The proxy accepts Stratum V1 connections, translates `mining.subscribe`, `mining.authorize`, and `mining.submit` into corresponding Pearl node JSON-RPC calls (`getmininginfo`, `getblocktemplate`, `submitblock`), and relays the results back.