package main

import (
	"testing"
)

func TestLoadConfigDefaults(t *testing.T) {
	config := loadConfig()

	if config.PearlNodeURL == "" {
		t.Error("PearlNodeURL should not be empty")
	}
	if config.StratumPort <= 0 {
		t.Errorf("StratumPort should be positive, got %d", config.StratumPort)
	}
	if config.MaxConnections <= 0 {
		t.Errorf("MaxConnections should be positive, got %d", config.MaxConnections)
	}
	if config.StratumDifficulty <= 0 {
		t.Errorf("StratumDifficulty should be positive, got %d", config.StratumDifficulty)
	}
}

func TestCoinConfigExists(t *testing.T) {
	coins := []string{"prl", "btc", "ltc", "ethw", "rvn"}
	for _, coin := range coins {
		cfg, ok := KryptexCoins[coin]
		if !ok {
			t.Errorf("KryptexCoins missing entry for %q", coin)
			continue
		}
		if cfg.Name == "" {
			t.Errorf("KryptexCoins[%q].Name should not be empty", coin)
		}
		if cfg.PoolHost == "" {
			t.Errorf("KryptexCoins[%q].PoolHost should not be empty", coin)
		}
		if cfg.PoolPort <= 0 {
			t.Errorf("KryptexCoins[%q].PoolPort should be positive, got %d", coin, cfg.PoolPort)
		}
	}
}

func TestDifficultyToTarget(t *testing.T) {
	client := &stratumClient{difficulty: 1}
	target := client.difficultyToTarget(1)
	if target == "" {
		t.Error("difficultyToTarget should return non-empty string")
	}

	// Higher difficulty should produce a smaller target
	targetLow := client.difficultyToTarget(1)
	targetHigh := client.difficultyToTarget(10)
	if targetLow <= targetHigh {
		t.Errorf("Higher difficulty should produce smaller target: low=%s high=%s", targetLow, targetHigh)
	}
}