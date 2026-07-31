package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"
)

type jsonRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int             `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
}

type jsonResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
}

type jsonRPCError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

type rpcClient struct {
	url      string
	user     string
	password string
	httpClient *http.Client
	requestID  int
}

func newRPCClient(url, user, password string) *rpcClient {
	return &rpcClient{
		url:      url,
		user:     user,
		password: password,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (c *rpcClient) call(method string, params interface{}) (json.RawMessage, error) {
	c.requestID++
	reqBody, err := json.Marshal(jsonRequest{
		JSONRPC: "1.0",
		ID:      c.requestID,
		Method:  method,
		Params:  mustMarshal(params),
	})
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequest("POST", c.url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "text/plain")
	httpReq.SetBasicAuth(c.user, c.password)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var jsonResp jsonResponse
	if err := json.Unmarshal(body, &jsonResp); err != nil {
		return nil, err
	}

	if jsonResp.Error != nil {
		return nil, &rpcError{
			code:    jsonResp.Error.Code,
			message: jsonResp.Error.Message,
		}
	}

	return jsonResp.Result, nil
}

func (c *rpcClient) getMiningInfo() (map[string]interface{}, error) {
	result, err := c.call("getmininginfo", nil)
	if err != nil {
		return nil, err
	}
	var info map[string]interface{}
	if err := json.Unmarshal(result, &info); err != nil {
		return nil, err
	}
	return info, nil
}

func (c *rpcClient) submitBlock(hex string) (string, error) {
	result, err := c.call("submitblock", []interface{}{hex})
	if err != nil {
		return "", err
	}
	if result != nil {
		var s string
		if err := json.Unmarshal(result, &s); err == nil {
			return s, nil
		}
	}
	return "accepted", nil
}

func (c *rpcClient) getBlockTemplate() (map[string]interface{}, error) {
	params := []interface{}{map[string]interface{}{"rules": []interface{}{"segwit"}}}
	result, err := c.call("getblocktemplate", params)
	if err != nil {
		return nil, err
	}
	var tpl map[string]interface{}
	if err := json.Unmarshal(result, &tpl); err != nil {
		return nil, err
	}
	return tpl, nil
}

func mustMarshal(v interface{}) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage("null")
	}
	return json.RawMessage(b)
}

type rpcError struct {
	code    int
	message string
}

func (e *rpcError) Error() string {
	return e.message
}