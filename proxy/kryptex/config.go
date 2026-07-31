package main

import "time"

type Config struct {
	PearlNodeURL     string        `env:"PEARLD_RPC_URL" default:"http://127.0.0.1:44107"`
	PearlNodeUser    string        `env:"PEARLD_RPC_USER" default:"rpcuser"`
	PearlNodePass    string        `env:"PEARLD_RPC_PASSWORD" default:"rpcpass"`
	StratumPort      int           `env:"STRATUM_PORT" default:"3333"`
	StratumTLS       bool          `env:"STRATUM_TLS" default:"false"`
	StratumCertFile  string        `env:"STRATUM_CERT_FILE" default:""`
	StratumKeyFile   string        `env:"STRATUM_KEY_FILE" default:""`
	HealthPort       int           `env:"HEALTH_PORT" default:"9900"`
	ReadTimeout      time.Duration `env:"READ_TIMEOUT" default:"30s"`
	WriteTimeout     time.Duration `env:"WRITE_TIMEOUT" default:"30s"`
	MaxConnections   int           `env:"MAX_CONNECTIONS" default:"1000"`
	StratumDifficulty int          `env:"STRATUM_DIFFICULTY" default:"1"`
}