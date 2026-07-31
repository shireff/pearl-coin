package main

type CoinConfig struct {
	Name        string
	PoolHost    string
	PoolPort    int
	WalletFmt   string
	Password    string
	Description string
}

var KryptexCoins = map[string]CoinConfig{
	"prl": {
		Name:        "Pearl (PRL)",
		PoolHost:    "prl.kryptex.network",
		PoolPort:    7048,
		WalletFmt:   "mining_username.worker_name",
		Password:    "x",
		Description: "Mine Pearl (PRL) via Kryptex Pool",
	},
	"btc": {
		Name:        "Bitcoin (BTC)",
		PoolHost:    "btc.kryptex.network",
		PoolPort:    7014,
		WalletFmt:   "mining_username.worker_name",
		Password:    "x",
		Description: "Mine Bitcoin via Kryptex Pool",
	},
	"ltc": {
		Name:        "Litecoin (LTC)",
		PoolHost:    "ltc.kryptex.network",
		PoolPort:    7016,
		WalletFmt:   "mining_username.worker_name",
		Password:    "x",
		Description: "Mine Litecoin via Kryptex Pool",
	},
	"ethw": {
		Name:        "EthereumPoW (ETHW)",
		PoolHost:    "ethw.kryptex.network",
		PoolPort:    7034,
		WalletFmt:   "mining_username.worker_name",
		Password:    "x",
		Description: "Mine EthereumPoW via Kryptex Pool",
	},
	"rvn": {
		Name:        "Ravencoin (RVN)",
		PoolHost:    "rvn.kryptex.network",
		PoolPort:    7031,
		WalletFmt:   "mining_username.worker_name",
		Password:    "x",
		Description: "Mine Ravencoin via Kryptex Pool",
	},
}