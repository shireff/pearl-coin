package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"
)

type server struct {
	config    *Config
	rpc       *rpcClient
	clientsMu sync.RWMutex
	clients   map[string]*stratumClient
	listener  net.Listener
	wg        sync.WaitGroup
	running   bool
}

func main() {
	config := loadConfig()

	s := &server{
		config:  &config,
		rpc:     newRPCClient(config.PearlNodeURL, config.PearlNodeUser, config.PearlNodePass),
		clients: make(map[string]*stratumClient),
	}

	if err := s.start(); err != nil {
		log.Fatalf("Failed to start Kryptex proxy: %v", err)
	}

	s.waitForShutdown()
}

func (s *server) start() error {
	var listener net.Listener
	var err error

	if s.config.StratumTLS {
		listener, err = s.startTLSServer()
	} else {
		listener, err = s.startPlainServer()
	}

	if err != nil {
		return fmt.Errorf("failed to start Stratum listener: %w", err)
	}

	s.listener = listener
	s.running = true

	s.log("info", "Kryptex Stratum proxy started",
		"coin", "prl",
		"port", s.config.StratumPort,
		"node", s.config.PearlNodeURL,
		"tls", s.config.StratumTLS,
	)

	s.wg.Add(1)
	go s.acceptLoop()

	s.wg.Add(1)
	go s.startHealthServer()

	return nil
}

func (s *server) startPlainServer() (net.Listener, error) {
	addr := fmt.Sprintf(":%d", s.config.StratumPort)
	return net.Listen("tcp", addr)
}

func (s *server) startTLSServer() (net.Listener, error) {
	cert, err := tls.LoadX509KeyPair(s.config.StratumCertFile, s.config.StratumKeyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load TLS certificate: %w", err)
	}

	pool := x509.NewCertPool()
	caCert, err := os.ReadFile(s.config.StratumCertFile)
	if err != nil {
		return nil, fmt.Errorf("failed to read CA cert: %w", err)
	}
	pool.AppendCertsFromPEM(caCert)

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", s.config.StratumPort))
	if err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientCAs:    pool,
		ClientAuth:   tls.VerifyClientCertIfGiven,
		MinVersion:   tls.VersionTLS12,
	}

	tlsListener := tls.NewListener(listener, tlsConfig)
	return tlsListener, nil
}

func (s *server) acceptLoop() {
	defer s.wg.Done()

	for {
		conn, err := s.listener.Accept()
		if err != nil {
			if s.running {
				s.log("error", "Accept error", "err", err.Error())
			}
			return
		}

		s.clientsMu.Lock()
		if len(s.clients) >= s.config.MaxConnections {
			s.clientsMu.Unlock()
			conn.Close()
			s.log("warn", "Connection rejected: max connections reached")
			continue
		}
		s.clientsMu.Unlock()

		go s.handleConnection(conn)
	}
}

func (s *server) handleConnection(conn net.Conn) {
	conn.SetDeadline(time.Now().Add(5 * time.Minute))

	client := newStratumClient(conn, s.config, "prl")

	s.clientsMu.Lock()
	s.clients[client.id] = client
	s.clientsMu.Unlock()

	defer func() {
		s.clientsMu.Lock()
		delete(s.clients, client.id)
		s.clientsMu.Unlock()
		conn.Close()
	}()

	client.serve()
}

func (s *server) startHealthServer() {
	defer s.wg.Done()

	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/status", s.handleStatus)

	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", s.config.HealthPort),
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}

	s.log("info", "Health server started", "port", s.config.HealthPort)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		s.log("error", "Health server error", "err", err.Error())
	}
}

func (s *server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"pool":   "kryptex",
		"coin":   "prl",
		"node":   s.config.PearlNodeURL,
	})
}

func (s *server) handleStatus(w http.ResponseWriter, r *http.Request) {
	s.clientsMu.RLock()
	clientCount := len(s.clients)
	s.clientsMu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"pool":           "kryptex",
		"coin":           "prl",
		"connected_nodes": clientCount,
		"node_url":       s.config.PearlNodeURL,
		"stratum_port":   s.config.StratumPort,
		"tls_enabled":    s.config.StratumTLS,
	})
}

func (s *server) waitForShutdown() {
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	<-sigChan
	s.log("info", "Shutdown signal received")
	s.stop()
}

func (s *server) stop() {
	s.running = false

	s.clientsMu.Lock()
	for _, client := range s.clients {
		client.disconnect()
	}
	s.clients = make(map[string]*stratumClient)
	s.clientsMu.Unlock()

	if s.listener != nil {
		s.listener.Close()
	}

	s.wg.Wait()
	s.log("info", "Kryptex proxy stopped")
}

func (s *server) log(level, msg string, keysAndVals ...interface{}) {
	logLine := fmt.Sprintf("[kryptex] [%s] %s", level, msg)
	for i := 0; i+1 < len(keysAndVals); i += 2 {
		logLine += fmt.Sprintf(" %s=%v", keysAndVals[i], keysAndVals[i+1])
	}
	log.Println(logLine)
}

func loadConfig() Config {
	config := Config{
		PearlNodeURL:      "http://127.0.0.1:44107",
		PearlNodeUser:     "rpcuser",
		PearlNodePass:     "rpcpass",
		StratumPort:       3333,
		StratumDifficulty:  1,
		HealthPort:         9900,
		MaxConnections:     1000,
	}

	v := os.Getenv("PEARLD_RPC_URL")
	if v != "" {
		config.PearlNodeURL = v
	}

	v = os.Getenv("PEARLD_RPC_USER")
	if v != "" {
		config.PearlNodeUser = v
	}

	v = os.Getenv("PEARLD_RPC_PASSWORD")
	if v != "" {
		config.PearlNodePass = v
	}

	v = os.Getenv("STRATUM_PORT")
	if v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			config.StratumPort = p
		}
	}

	v = os.Getenv("STRATUM_TLS")
	if v == "true" {
		config.StratumTLS = true
	}

	v = os.Getenv("STRATUM_CERT_FILE")
	if v != "" {
		config.StratumCertFile = v
	}

	v = os.Getenv("STRATUM_KEY_FILE")
	if v != "" {
		config.StratumKeyFile = v
	}

	v = os.Getenv("HEALTH_PORT")
	if v != "" {
		if p, err := strconv.Atoi(v); err == nil {
			config.HealthPort = p
		}
	}

	v = os.Getenv("MAX_CONNECTIONS")
	if v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			config.MaxConnections = n
		}
	}

	v = os.Getenv("STRATUM_DIFFICULTY")
	if v != "" {
		if d, err := strconv.Atoi(v); err == nil {
			config.StratumDifficulty = d
		}
	}

	return config
}