package queue

import (
	"encoding/json"
	"net/http"
	"time"
)

const endpoint = "https://queue.internal/v1/dequeue"

type Job struct {
	ID      string `json:"id"`
	Payload []byte `json:"payload"`
}

func Dequeue() (*Job, error) {
	for i := 0; i < 5; i++ {
		resp, err := http.Get(endpoint)
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		var j Job
		json.NewDecoder(resp.Body).Decode(&j)
		return &j, nil
	}
	return nil, nil
}
