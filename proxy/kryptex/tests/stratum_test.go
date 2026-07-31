package main

import (
	"net"
	"testing"
	"time"
)

func TestNewStratumClient(t *testing.T) {
	config := &Config{
		StratumPort:      3333,
		StratumDifficulty: 1,
		MaxConnections:    1000,
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create test listener: %v", err)
	}
	defer ln.Close()

	go func() {
		conn, _ := ln.Accept()
		if conn != nil {
			conn.Close()
		}
	}()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Failed to dial test listener: %v", err)
	}
	defer conn.Close()

	client := newStratumClient(conn, config, "prl")
	if client == nil {
		t.Fatal("Expected non-nil stratumClient")
	}
	if client.coinKey != "prl" {
		t.Errorf("Expected coinKey 'prl', got %q", client.coinKey)
	}
	if client.difficulty != 1 {
		t.Errorf("Expected difficulty 1, got %d", client.difficulty)
	}
	if client.id == "" {
		t.Error("Expected non-empty client id")
	}
}

func TestStratumClientDisconnect(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Failed to create test listener: %v", err)
	}
	defer ln.Close()

	go func() {
		conn, _ := ln.Accept()
		if conn != nil {
			time.Sleep(50 * time.Millisecond)
			conn.Close()
		}
	}()

	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatalf("Failed to dial test listener: %v", err)
	}

	client := newStratumClient(conn, &Config{StratumPort: 3333}, "prl")

	done := make(chan struct{})
	go func() {
		client.handle()
		close(done)
	}()

	select {
	case <-done:
		t.Log("Client disconnected as expected")
	case <-time.After(3 * time.Second):
		t.Error("Client handle did not return after disconnect")
	}
}