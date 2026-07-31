package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

type stratumClient struct {
	conn       net.Conn
	reader     *bufio.Reader
	writer     *bufio.Writer
	config     *Config
	rpc        *rpcClient
	coinKey    string
	wallet     string
	worker     string
	difficulty int
	id         string
	mu         sync.Mutex
	connected  bool
	subscribed bool
	authorized bool
}

type stratumRequest struct {
	ID     json.RawMessage       `json:"id"`
	Method string                `json:"method"`
	Params []json.RawMessage     `json:"params"`
}

type stratumResponse struct {
	ID     json.RawMessage `json:"id"`
	Result interface{}     `json:"result,omitempty"`
	Error  *stratumError   `json:"error,omitempty"`
}

type stratumError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type miningNotify struct {
	JobID        string `json:"job_id"`
	Blob         string `json:"blob"`
	Target       string `json:"target"`
	Algorithm    string `json:"algorithm"`
	Coin         string `json:"coin"`
	Height       int64  `json:"height"`
	CleanJobs    bool   `json:"clean_jobs"`
}

type submitRequest struct {
	JobID    string `json:"job_id"`
	Nonce    string `json:"nonce"`
	Result   string `json:"result"`
	Worker   string `json:"worker"`
}

func newStratumClient(conn net.Conn, config *Config, coinKey string) *stratumClient {
	return &stratumClient{
		conn:       conn,
		reader:     bufio.NewReader(conn),
		writer:     bufio.NewWriter(conn),
		config:     config,
		rpc:        newRPCClient(config.PearlNodeURL, config.PearlNodeUser, config.PearlNodePass),
		coinKey:    coinKey,
		difficulty: config.StratumDifficulty,
		id:         uuid.New().String(),
		connected:  true,
	}
}

func (c *stratumClient) handle() {
	defer c.disconnect()

	c.log("info", "Stratum client connected", "coin", c.coinKey)

	for {
		line, err := c.reader.ReadString('\n')
		if err != nil {
			if err != io.EOF {
				c.log("warn", "Read error", "err", err.Error())
			}
			return
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if err := c.handleMessage(line); err != nil {
			c.log("error", "Message handling error", "err", err.Error())
			return
		}
	}
}

func (c *stratumClient) handleMessage(raw string) error {
	var req stratumRequest
	if err := json.Unmarshal([]byte(raw), &req); err != nil {
		c.sendError(nil, -32700, "Parse error")
		return fmt.Errorf("parse error: %w", err)
	}

	c.log("debug", "Received request", "method", req.Method, "id", string(req.ID))

	switch req.Method {
	case "mining.subscribe":
		return c.handleSubscribe(req.ID, req.Params)
	case "mining.authorize":
		return c.handleAuthorize(req.ID, req.Params)
	case "mining.submit":
		return c.handleSubmit(req.ID, req.Params)
	case "mining.extranonce.subscribe":
		return c.sendResponse(req.ID, true)
	case "mining.set_difficulty":
		return c.handleSetDifficulty(req.ID, req.Params)
	case "mining.set_extranonce":
		return c.sendResponse(req.ID, true)
	default:
		return c.sendError(req.ID, -32601, "Method not found")
	}
}

func (c *stratumClient) handleSubscribe(id json.RawMessage, params []json.RawMessage) error {
	extranonce := uuid.New().String()[:8]

	subscribeResult := []interface{}{
		[]interface{}{
			[]string{"mining.set_difficulty"},
			extranonce,
		},
		[]interface{}{
			[]string{"mining.notify"},
			extranonce,
		},
	}

	c.subscribed = true
	c.log("info", "Client subscribed")
	return c.sendResponse(id, subscribeResult)
}

func (c *stratumClient) handleAuthorize(id json.RawMessage, params []json.RawMessage) error {
	if len(params) < 1 {
		return c.sendError(id, -32602, "Missing worker name")
	}

	var worker string
	if err := json.Unmarshal(params[0], &worker); err != nil {
		return c.sendError(id, -32602, "Invalid worker name")
	}

	parts := strings.SplitN(worker, ".", 2)
	if len(parts) < 2 {
		return c.sendError(id, -32602, "Worker must be in format username.workername")
	}

	c.wallet = parts[0]
	c.worker = parts[1]
	c.authorized = true

	c.log("info", "Client authorized", "wallet", c.wallet, "worker", c.worker)
	return c.sendResponse(id, true)
}

func (c *stratumClient) handleSubmit(id json.RawMessage, params []json.RawMessage) error {
	if !c.authorized {
		return c.sendError(id, -32603, "Not authorized")
	}

	if len(params) < 4 {
		return c.sendError(id, -32602, "Missing submit parameters")
	}

	var submit submitRequest
	submit.JobID = string(params[0])
	submit.Nonce = string(params[1])
	submit.Result = string(params[2])
	submit.Worker = string(params[3])

	c.log("info", "Submit received", "job", submit.JobID, "nonce", submit.Nonce)

	go c.processSubmission(submit)

	return c.sendResponse(id, true)
}

func (c *stratumClient) processSubmission(submit submitRequest) {
	c.log("debug", "Processing submission", "job", submit.JobID)

	blockHex, err := c.buildBlockFromSubmission(submit)
	if err != nil {
		c.log("error", "Failed to build block", "err", err.Error())
		return
	}

	result, err := c.rpc.submitBlock(blockHex)
	if err != nil {
		c.log("error", "Block submission failed", "err", err.Error())
		return
	}

	c.log("info", "Block submission result", "result", result)
}

func (c *stratumClient) buildBlockFromSubmission(submit submitRequest) (string, error) {
	template, err := c.rpc.getBlockTemplate()
	if err != nil {
		return "", fmt.Errorf("failed to get block template: %w", err)
	}

	coinbaseTx, ok := template["coinbasevalue"].(string)
	if !ok {
		coinbaseTx = "0x00"
	}

	prevHash, ok := template["previousblockhash"].(string)
	if !ok {
		prevHash = ""
	}

	height := int64(0)
	if h, ok := template["height"].(float64); ok {
		height = int64(h)
	}

	targetDiff := template["bits"].(string)

	blockBlob := fmt.Sprintf(
		"%s%s%s%s%s",
		prevHash,
		coinbaseTx,
		submit.Nonce,
		submit.Result,
		strconv.FormatInt(height, 10),
	)

	blockHeader := c.constructBlockHeader(template, submit, height)
	return blockHeader, nil
}

func (c *stratumClient) constructBlockHeader(template map[string]interface{}, submit submitRequest, height int64) string {
	version := uint32(536870912)

	prevHashHex := ""
	if ph, ok := template["previousblockhash"].(string); ok && ph != "" {
		prevHashHex = ph
	}

	merkleRoot := "0000000000000000000000000000000000000000000000000000000000000000"
	if mr, ok := template["merkleroot"].(string); ok {
		merkleRoot = mr
	}

	timeStamp := uint32(time.Now().Unix())
	if ts, ok := template["curtime"].(float64); ok {
		timeStamp = uint32(ts)
	}

	bits := ""
	if b, ok := template["bits"].(string); ok {
		bits = b
	}

	nonceHex := submit.Nonce
	if n, err := strconv.ParseUint(nonceHex, 16, 32); err == nil {
		nonceHex = fmt.Sprintf("%08x", uint32(n))
	}

	header := fmt.Sprintf(
		"%08x%064x%064x%08x%08x%08x",
		version,
		decodeHex(prevHashHex),
		decodeHex(merkleRoot),
		timeStamp,
		decodeHex(bits),
		uint32(n),
	)

	return header
}

func decodeHex(s string) uint64 {
	if s == "" {
		return 0
	}
	v, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return 0
	}
	return v
}

func (c *stratumClient) handleSetDifficulty(id json.RawMessage, params []json.RawMessage) error {
	if len(params) < 1 {
		return c.sendError(id, -32602, "Missing difficulty value")
	}

	var diff float64
	if err := json.Unmarshal(params[0], &diff); err != nil {
		return c.sendError(id, -32602, "Invalid difficulty value")
	}

	c.difficulty = int(diff)
	c.log("info", "Difficulty updated", "difficulty", c.difficulty)
	return c.sendResponse(id, true)
}

func (c *stratumClient) sendJobNotify() {
	if !c.authorized {
		return
	}

	template, err := c.rpc.getBlockTemplate()
	if err != nil {
		c.log("error", "Failed to get block template for notify", "err", err.Error())
		return
	}

	jobID := uuid.New().String()[:8]
	blob := c.buildBlob(template)
	target := c.difficultyToTarget(c.difficulty)

	height := int64(0)
	if h, ok := template["height"].(float64); ok {
		height = int64(h)
	}

	notify := miningNotify{
		JobID:     jobID,
		Blob:      blob,
		Target:    target,
		Algorithm: "sha256d",
		Coin:      c.coinKey,
		Height:    height,
		CleanJobs: true,
	}

	if err := c.sendNotification("mining.notify", notify); err != nil {
		c.log("error", "Failed to send job notification", "err", err.Error())
	}
}

func (c *stratumClient) buildBlob(template map[string]interface{}) string {
	prevHash := ""
	if ph, ok := template["previousblockhash"].(string); ok {
		prevHash = ph
	}

	coinbase := ""
	if cv, ok := template["coinbasevalue"].(string); ok {
		coinbase = cv
	}

	bits := ""
	if b, ok := template["bits"].(string); ok {
		bits = b
	}

	return fmt.Sprintf("%s%s%s", prevHash, coinbase, bits)
}

func (c *stratumClient) difficultyToTarget(difficulty int) string {
	target := new(big.Int).Lsh(big.NewInt(1), 256-int32(difficulty))
	target.Sub(target, big.NewInt(1))
	return fmt.Sprintf("%064x", target)
}

func (c *stratumClient) sendResponse(id json.RawMessage, result interface{}) error {
	resp := stratumResponse{
		ID:     id,
		Result: result,
	}
	return c.sendJSON(resp)
}

func (c *stratumClient) sendError(id json.RawMessage, code int, message string) error {
	resp := stratumResponse{
		ID:    id,
		Error: &stratumError{Code: code, Message: message},
	}
	return c.sendJSON(resp)
}

func (c *stratumClient) sendNotification(method string, params interface{}) error {
	msg := map[string]interface{}{
		"method": method,
		"params": params,
	}
	return c.sendJSON(msg)
}

func (c *stratumClient) sendJSON(v interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	_, err = c.writer.Write(data)
	if err != nil {
		return err
	}
	return c.writer.Flush()
}

func (c *stratumClient) disconnect() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected {
		c.connected = false
		c.log("info", "Client disconnected")
		c.conn.Close()
	}
}

func (c *stratumClient) log(level, msg string, keysAndVals ...interface{}) {
	logLine := fmt.Sprintf("[stratum] [%s] %s", level, msg)
	for i := 0; i+1 < len(keysAndVals); i += 2 {
		logLine += fmt.Sprintf(" %s=%v", keysAndVals[i], keysAndVals[i+1])
	}
	fmt.Println(logLine)
}

func (c *stratumClient) runNotifyLoop(stopCh <-chan struct{}) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-stopCh:
			return
		case <-ticker.C:
			if c.connected && c.authorized {
				c.sendJobNotify()
			}
		}
	}
}

func (c *stratumClient) serve() {
	stopCh := make(chan struct{})
	defer close(stopCh)

	go c.runNotifyLoop(stopCh)

	c.sendJobNotify()
	c.handle()
}