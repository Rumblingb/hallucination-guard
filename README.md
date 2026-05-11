# Hallucination Guard API

> **Detect hallucinations in agent claims.** Verify what your AI agents say against ground-truth context, catch hallucinations before they reach your users.

**Pricing:** $99/month — [Subscribe now](https://buy.stripe.com/3cs4hT7EPc0lcZy5kl)

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set your OpenRouter key (optional — falls back to heuristic)
export OPENROUTER_API_KEY="sk-or-..."
export HALLU_FREE_KEYS="jsk_your_dev_key_here"

# 3. Run
uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Verify a claim
curl -X POST http://localhost:8000/v1/verify \
  -H "Authorization: Bearer jsk_your_dev_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "The Eiffel Tower is in Berlin",
    "context": "The Eiffel Tower is located in Paris, France. It was built in 1889."
  }'
```

---

## Table of Contents

- [Overview](#overview)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /v1/verify](#post-v1verify)
  - [POST /v1/verify-batch](#post-v1verify-batch)
  - [GET /health](#get-health)
- [Response Format](#response-format)
- [Strictness Levels](#strictness-levels)
- [Verification Methods](#verification-methods)
- [Rate Limits](#rate-limits)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Error Handling](#error-handling)
- [Examples](#examples)
- [Pricing](#pricing)
- [Support](#support)

---

## Overview

The Hallucination Guard API helps you build trustworthy AI agents by checking every claim against its source context before delivery. It uses a two-tier verification system:

1. **LLM-as-Judge (primary):** Sends the claim + context to an LLM (via OpenRouter) with a structured prompt that returns a JSON verdict including a hallucination score, supported flag, explanation, and evidence.
2. **Heuristic Fallback (automatic):** If no OpenRouter key is configured or the LLM call fails, we fall back to a keyword-overlap Jaccard similarity scorer that's fast, zero-cost, and still useful.

---

## Authentication

Every request requires an API key with the `jsk_` prefix. Pass it via the `Authorization` header:

```
Authorization: Bearer jsk_your_api_key_here
```

Or via the `X-API-Key` header:

```
X-API-Key: jsk_your_api_key_here
```

### Managing Keys

Keys are configured via environment variables:

```bash
# Free keys (1000 req/day each)
export HALLU_FREE_KEYS="jsk_key1,jsk_key2"

# Pro keys (unlimited)
export HALLU_PRO_KEYS="jsk_pro_key1,jsk_pro_key2"
```

If no keys are set, a single development key is auto-generated on startup and printed to the console.

---

## Endpoints

### POST /v1/verify

Verify a single agent claim against its context.

**Request:**

```json
{
  "claim": "string (1-10,000 chars, required)",
  "context": "string (1-50,000 chars, required)",
  "strictness": "string | null ('strict', 'moderate', 'lenient')"
}
```

**Response (200):**

```json
{
  "score": 0.15,
  "supported": true,
  "explanation": "Claim appears well-supported by context (87% of claim keywords found in context).",
  "evidence": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
  "method": "llm_judge"
}
```

**cURL:**

```bash
curl -X POST https://api.hallucination-guard.nousresearch.com/v1/verify \
  -H "Authorization: Bearer jsk_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "claim": "Einstein won the Nobel Prize in 1921",
    "context": "Albert Einstein received the Nobel Prize in Physics in 1921 for his work on the photoelectric effect."
  }'
```

---

### POST /v1/verify-batch

Verify up to 100 claims in a single request. Each item is independent.

**Request:**

```json
{
  "claims": [
    {
      "claim": "Paris is the capital of France",
      "context": "Paris is the capital and most populous city of France."
    },
    {
      "claim": "The moon is made of cheese",
      "context": "The Moon is Earth's only natural satellite, composed primarily of silicate rock."
    }
  ]
}
```

**Response (200):**

```json
{
  "results": [
    {
      "score": 0.05,
      "supported": true,
      "explanation": "...",
      "evidence": "...",
      "method": "llm_judge"
    },
    {
      "score": 0.92,
      "supported": false,
      "explanation": "...",
      "evidence": "...",
      "method": "llm_judge"
    }
  ]
}
```

**cURL:**

```bash
curl -X POST https://api.hallucination-guard.nousresearch.com/v1/verify-batch \
  -H "Authorization: Bearer jsk_your_key" \
  -H "Content-Type: application/json" \
  -d '{"claims": [{"claim": "...", "context": "..."}]}'
```

---

### GET /health

Simple health check endpoint.

**Response:**

```json
{
  "status": "ok",
  "api_keys_configured": 3,
  "openrouter_configured": true,
  "version": "1.0.0"
}
```

**cURL:**

```bash
curl https://api.hallucination-guard.nousresearch.com/health
```

---

## Response Format

| Field         | Type    | Description |
|---------------|---------|-------------|
| `score`       | float   | Hallucination score from 0.0 (fully supported) to 1.0 (fully hallucinated) |
| `supported`   | bool    | `true` if the claim is supported by context (score < 0.5) |
| `explanation` | string  | Human-readable explanation of the verdict |
| `evidence`    | string  | Relevant excerpt(s) from the context that justify the verdict |
| `method`      | string  | `"llm_judge"` or `"heuristic_fallback"` |

---

## Strictness Levels

| Level      | Description |
|------------|-------------|
| `strict`   | Even minor discrepancies increase the score. Use for high-stakes applications (medical, legal, financial). |
| `moderate` | Default. Balances precision with recall. |
| `lenient`  | Only flag clear contradictions. Use for creative or brainstorming contexts. |

---

## Verification Methods

### 1. LLM-as-Judge (primary)

- Calls OpenRouter with the `meta-llama/llama-3.2-3b-instruct:free` model (configurable).
- Uses a structured JSON output prompt with temperature 0.1 for deterministic results.
- Returns precise, nuanced verdicts with cited evidence.
- Requires `OPENROUTER_API_KEY` environment variable.

### 2. Heuristic Fallback (automatic)

- Jaccard similarity over keyword sets (words ≥ 3 characters, lowercased).
- Extracts the most relevant sentences from context for the `evidence` field.
- Adjusts score by ±30% based on strictness level.
- No external dependencies, always available, zero latency.

---

## Rate Limits

| Tier  | Daily Limit | Use Case |
|-------|-------------|----------|
| Free  | 1,000 req/day | Testing, development, low-volume |
| Pro   | Unlimited    | Production, enterprise |

Free tier keys return HTTP 429 when the limit is reached. Pro keys have no cap.

---

## Environment Variables

| Variable              | Default                                | Description |
|-----------------------|----------------------------------------|-------------|
| `OPENROUTER_API_KEY`  | `""` (fallback only)                   | OpenRouter API key for LLM judge |
| `OPENROUTER_MODEL`    | `meta-llama/llama-3.2-3b-instruct:free` | OpenRouter model to use |
| `OPENROUTER_TIMEOUT`  | `15`                                   | Timeout in seconds for OpenRouter calls |
| `HALLU_FREE_KEYS`     | `""` (auto-generated dev key)          | Comma-separated free-tier API keys |
| `HALLU_PRO_KEYS`      | `""`                                   | Comma-separated pro-tier API keys |
| `FREE_DAILY_LIMIT`    | `1000`                                 | Daily request limit for free-tier keys |
| `PORT`                | `8000`                                 | HTTP server port |
| `RELOAD`              | `0`                                    | Enable auto-reload on file changes (development) |

---

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  hallucination-guard:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - HALLU_PRO_KEYS=${HALLU_PRO_KEYS}
      - HALLU_FREE_KEYS=${HALLU_FREE_KEYS}
```

### Production Considerations

- Run behind a reverse proxy (nginx, Caddy) for TLS termination.
- Set `OPENROUTER_TIMEOUT` to match your application's tolerance (default 15s).
- Use Pro-tier keys in production for unlimited throughput.
- Monitor OpenRouter rate limits and latency.

---

## Error Handling

| Status Code | Error Code | Description |
|-------------|-----------|-------------|
| 200         | —          | Success |
| 400         | —          | Invalid request body (validation error) |
| 401         | 401        | Missing or invalid API key |
| 429         | 429        | Rate limit exceeded (free tier) |
| 500         | 500        | Internal server error |

Error responses follow the format:

```json
{
  "detail": "Missing API key. Provide via 'Authorization: Bearer jsk_...' or 'X-API-Key: jsk_...' header.",
  "code": 401
}
```

---

## Examples

### Python

```python
import httpx

API_KEY = "jsk_your_key"
BASE_URL = "http://localhost:8000"

client = httpx.Client(
    base_url=BASE_URL,
    headers={"Authorization": f"Bearer {API_KEY}"},
)

# Single verification
resp = client.post("/v1/verify", json={
    "claim": "Water boils at 100°C at sea level",
    "context": "At standard atmospheric pressure (sea level), water boils at 100 degrees Celsius.",
})
print(resp.json())
# => {"score": 0.02, "supported": true, ...}

# Batch verification
resp = client.post("/v1/verify-batch", json={
    "claims": [
        {"claim": "...", "context": "..."},
        {"claim": "...", "context": "..."},
    ],
})
print(resp.json())
```

### JavaScript / Node.js

```javascript
const response = await fetch("http://localhost:8000/v1/verify", {
  method: "POST",
  headers: {
    "Authorization": "Bearer jsk_your_key",
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    claim: "The sun revolves around the Earth",
    context: "Earth orbits the Sun at an average distance of 149.6 million kilometers.",
  }),
});
const data = await response.json();
console.log(data);
// => {"score": 0.88, "supported": false, ...}
```

### Evaluation Pipeline Script

```python
import httpx, json

client = httpx.Client(
    base_url="http://localhost:8000",
    headers={"Authorization": "Bearer jsk_your_key"},
)

test_cases = [
    {"claim": "Paris is in France", "context": "Paris is the capital of France.", "expected": True},
    {"claim": "Paris is in Italy", "context": "Paris is the capital of France.", "expected": False},
]

passed = 0
for tc in test_cases:
    resp = client.post("/v1/verify", json={"claim": tc["claim"], "context": tc["context"]})
    result = resp.json()
    correct = result["supported"] == tc["expected"]
    print(f"{'✓' if correct else '✗'} score={result['score']:.2f} supported={result['supported']} | {tc['claim']}")
    if correct:
        passed += 1

print(f"\nAccuracy: {passed}/{len(test_cases)} ({passed/len(test_cases)*100:.0f}%)")
```

---

## Pricing

| Plan     | Price     | Requests/day | Features |
|----------|-----------|-------------|----------|
| Free     | $0        | 1,000       | LLM judge + heuristic fallback, community support |
| **Pro**  | **$99/mo**| Unlimited   | Priority routing, SLA, dedicated support |

**[Subscribe to Pro →](https://buy.stripe.com/3cs4hT7EPc0lcZy5kl)**

Pro subscribers get:
- Unlimited requests per month
- Higher rate limits on OpenRouter models
- Priority support via email
- Early access to new features (custom models, streaming, webhook callbacks)

---

## Support

- **Email:** support@nousresearch.com
- **Docs:** https://hallucination-guard.nousresearch.com/docs
- **Status:** https://status.nousresearch.com

---

*Built by Nous Research — making AI agents trustworthy.*
