"""
Hallucination Guard API
=======================
REST API that checks agent claims against context to detect hallucinations.
Uses LLM-as-judge via OpenRouter (free models) with fallback to keyword-overlap heuristic.
"""

import os
import re
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, status, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")
OPENROUTER_TIMEOUT = int(os.environ.get("OPENROUTER_TIMEOUT", "15"))

# In-memory API key store: key -> {"tier": "free"|"pro", "rate_limit_reset": datetime, "usage_today": int}
# Free keys: 1000 req/day. Pro keys: unlimited.
API_KEYS = {}

# Populate from environment: comma-separated lists of keys.
# Format: HALLU_FREE_KEYS=sk_test_xxx,sk_test_yyy  HALLU_PRO_KEYS=sk_pro_xxx,sk_pro_yyy
_free_keys_raw = os.environ.get("HALLU_FREE_KEYS", "").strip()
_pro_keys_raw = os.environ.get("HALLU_PRO_KEYS", "").strip()
# If none set, create a default dev key
if not _free_keys_raw and not _pro_keys_raw:
    from secrets import token_hex
    _dev_key = f"jsk_{token_hex(16)}"
    API_KEYS[_dev_key] = {"tier": "free", "usage_today": 0, "rate_limit_reset": datetime.utcnow()}
    print(f"[startup] No API keys configured via env. Generated dev key: {_dev_key}")
else:
    for _k in _free_keys_raw.split(","):
        _k = _k.strip()
        if _k:
            API_KEYS[_k] = {"tier": "free", "usage_today": 0, "rate_limit_reset": datetime.utcnow()}
    for _k in _pro_keys_raw.split(","):
        _k = _k.strip()
        if _k:
            API_KEYS[_k] = {"tier": "pro", "usage_today": 0, "rate_limit_reset": datetime.utcnow()}

FREE_DAILY_LIMIT = int(os.environ.get("FREE_DAILY_LIMIT", "1000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("hallucination-guard")

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class VerifyRequest(BaseModel):
    claim: str = Field(..., description="The agent claim to verify", min_length=1, max_length=10000)
    context: str = Field(..., description="The ground-truth context to check against", min_length=1, max_length=50000)
    strictness: Optional[str] = Field(None, description="Strictness level: 'strict', 'moderate', or 'lenient'")


class VerifyResponse(BaseModel):
    score: float = Field(..., description="Hallucination score 0.0=supported .. 1.0=hallucinated")
    supported: bool = Field(..., description="Whether the claim is supported by the context")
    explanation: str = Field(..., description="Human-readable explanation of the verdict")
    evidence: str = Field(..., description="Relevant excerpt(s) from context that support or refute the claim")
    method: str = Field(..., description="Method used: 'llm_judge' or 'heuristic_fallback'")


class BatchVerifyItem(BaseModel):
    claim: str = Field(..., min_length=1, max_length=10000)
    context: str = Field(..., min_length=1, max_length=50000)
    strictness: Optional[str] = None


class BatchVerifyRequest(BaseModel):
    claims: list[BatchVerifyItem] = Field(..., min_length=1, max_length=100)


class BatchVerifyResponse(BaseModel):
    results: list[VerifyResponse]


class HealthResponse(BaseModel):
    status: str
    api_keys_configured: int
    openrouter_configured: bool
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Lifespan & app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Hallucination Guard API starting up")
    log.info(f"  API keys loaded: {len(API_KEYS)} ({sum(1 for v in API_KEYS.values() if v['tier']=='free')} free, {sum(1 for v in API_KEYS.values() if v['tier']=='pro')} pro)")
    log.info(f"  OpenRouter: {'configured' if OPENROUTER_API_KEY else 'NOT configured (will use heuristic fallback)'}")
    yield
    log.info("Hallucination Guard API shutting down")


app = FastAPI(
    title="Hallucination Guard API",
    description="Detect hallucinations in agent claims by verifying them against ground-truth context.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for ease of use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def _check_rate_limit(key_info: dict):
    """Check and update rate limit for a free-tier key. Pro keys are unlimited."""
    if key_info["tier"] == "pro":
        return  # unlimited

    now = datetime.utcnow()
    reset = key_info.get("rate_limit_reset")

    # Reset counter if a new day has started
    if reset and (now - reset) > timedelta(days=1):
        key_info["usage_today"] = 0
        key_info["rate_limit_reset"] = now

    if key_info["usage_today"] >= FREE_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Daily rate limit ({FREE_DAILY_LIMIT} requests) exceeded. "
                   f"Upgrade to Pro at https://hallucination-guard.nousresearch.com/pricing "
                   f"or wait for next reset.",
        )

    key_info["usage_today"] += 1


def _extract_bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str | None:
    """Return the raw token from the Authorization header, stripping any scheme prefix
    (Bearer, Basic, etc.) so we handle cases where the client sends 'jsk_xxx' directly
    after 'Bearer '."""
    if credentials is None:
        return None
    # HTTPBearer already parses the token after "Bearer "
    return credentials.credentials


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_auth(request: Request, credentials: HTTPAuthorizationCredentials | None = Security(security)) -> dict:
    """Dependency: validate the API key and return its info dict, or raise 401."""
    raw_token = _extract_bearer_token(credentials)

    # Also check X-API-Key header as fallback
    if not raw_token:
        raw_token = request.headers.get("X-API-Key", "").strip()

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via 'Authorization: Bearer jsk_...' or 'X-API-Key: jsk_...' header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if raw_token not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Ensure you're using a valid jsk_-prefixed key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    key_info = API_KEYS[raw_token]
    _check_rate_limit(key_info)
    return key_info


# ---------------------------------------------------------------------------
# LLM Judge (OpenRouter)
# ---------------------------------------------------------------------------

_LLM_JUDGE_SYSTEM_PROMPT = """You are a hallucination detection expert. Your job is to determine whether an agent's CLAIM is supported by a provided CONTEXT.

Analyze the claim against the context and return a JSON object with these fields:
- "score": float between 0.0 (fully supported) and 1.0 (fully hallucinated)
- "supported": boolean — true if the claim is supported by the context
- "explanation": short human-readable explanation of your verdict
- "evidence": the exact quote(s) from the context that support or refute the claim

Guidelines:
- A score of 0.0-0.3 means the claim is well-supported.
- A score of 0.4-0.6 means partial support — some parts match, some don't.
- A score of 0.7-1.0 means the claim is mostly or entirely hallucinated.
- Minor factual discrepancies should raise the score but not necessarily flip to "not supported".
- If the claim introduces information not present in the context, that is hallucination.
- Only return valid JSON, no markdown fences, no extra text."""


async def _llm_judge(claim: str, context: str, strictness: str | None) -> dict | None:
    """Call OpenRouter with structured output. Returns parsed JSON or None on failure."""
    if not OPENROUTER_API_KEY:
        return None

    strictness_instruction = ""
    if strictness:
        strictness_instruction = f"\nStrictness level: {strictness}. "
        if strictness == "strict":
            strictness_instruction += "Be very strict — even minor discrepancies should increase the score."
        elif strictness == "lenient":
            strictness_instruction += "Be lenient — only flag clear contradictions."
        else:
            strictness_instruction += "Use default moderate strictness."

    user_prompt = f"""CLAIM:
{claim}

CONTEXT:
{context}
{strictness_instruction}

Does the claim match the context? Return JSON with score, supported, explanation, and evidence."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://hallucination-guard.nousresearch.com",
        "X-Title": "Hallucination Guard API",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": _LLM_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }

    try:
        async with httpx.AsyncClient(timeout=OPENROUTER_TIMEOUT) as client:
            resp = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
        if resp.status_code != 200:
            log.warning(f"OpenRouter returned {resp.status_code}: {resp.text[:300]}")
            return None

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        # Parse the JSON response
        import json as _json
        result = _json.loads(content)

        # Validate required fields
        for field in ("score", "supported", "explanation", "evidence"):
            if field not in result:
                raise ValueError(f"Missing field '{field}' in LLM response")

        result["score"] = float(result["score"])
        result["score"] = max(0.0, min(1.0, result["score"]))
        result["supported"] = bool(result["supported"])
        result["explanation"] = str(result["explanation"])
        result["evidence"] = str(result["evidence"])

        return result

    except Exception as exc:
        log.warning(f"LLM judge failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Heuristic fallback scorer (keyword overlap)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    """Lowercase, split on non-alpha, return set of words (min 3 chars)."""
    text = text.lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return {w for w in words if len(w) >= 3}


def _extract_key_snippets(context: str, claim_words: set[str], max_chars: int = 500) -> str:
    """Find the most relevant sentences from context for evidence display."""
    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []
    for s in sentences:
        s_words = _tokenize(s)
        overlap = len(s_words & claim_words)
        if overlap > 0:
            scored.append((overlap, s.strip()))

    scored.sort(key=lambda x: -x[0])
    evidence_parts = []
    total = 0
    for _, sent in scored:
        if total + len(sent) > max_chars:
            break
        evidence_parts.append(sent)
        total += len(sent)
    return " ".join(evidence_parts) if evidence_parts else "(no relevant evidence found)"


def _heuristic_score(claim: str, context: str, strictness: str | None = None) -> dict:
    """
    Keyword-overlap heuristic scorer.

    Algorithm:
    1. Tokenize both claim and context into word sets (min 3 chars, lowercased).
    2. Compute Jaccard similarity = |intersection| / |union|.
    3. Hallucination score ≈ 1 - Jaccard (higher = more hallucinated).
    4. Adjust for strictness: strict penalises harder, lenient is more forgiving.
    """
    claim_words = _tokenize(claim)
    context_words = _tokenize(context)

    if not claim_words:
        return {
            "score": 0.5,
            "supported": False,
            "explanation": "Claim contains no meaningful words to evaluate.",
            "evidence": "",
            "method": "heuristic_fallback",
        }

    if not context_words:
        return {
            "score": 1.0,
            "supported": False,
            "explanation": "Context is empty — no basis to verify the claim.",
            "evidence": "",
            "method": "heuristic_fallback",
        }

    intersection = claim_words & context_words
    union = claim_words | context_words
    jaccard = len(intersection) / len(union)

    # Base hallucination score: 1 - Jaccard
    score = 1.0 - jaccard

    # Adjust for strictness
    if strictness == "strict":
        score = min(1.0, score * 1.3)
    elif strictness == "lenient":
        score = max(0.0, score * 0.7)

    # Clamp
    score = max(0.0, min(1.0, score))

    supported = score < 0.5
    evidence = _extract_key_snippets(context, claim_words)

    # Build a human-readable explanation
    overlap_pct = (len(intersection) / len(claim_words) * 100) if claim_words else 0
    if score < 0.3:
        explanation = (
            f"Claim appears well-supported by context "
            f"({overlap_pct:.0f}% of claim keywords found in context)."
        )
    elif score < 0.5:
        explanation = (
            f"Claim is partially supported "
            f"({overlap_pct:.0f}% of claim keywords found in context). "
            f"Some elements may not be fully grounded."
        )
    elif score < 0.7:
        explanation = (
            f"Claim may be partially hallucinated "
            f"(only {overlap_pct:.0f}% of claim keywords found in context). "
            f"Significant portions lack supporting evidence."
        )
    else:
        explanation = (
            f"Claim appears largely hallucinated "
            f"(only {overlap_pct:.0f}% of claim keywords found in context). "
            f"The context does not adequately support this claim."
        )

    return {
        "score": round(score, 4),
        "supported": supported,
        "explanation": explanation,
        "evidence": evidence,
        "method": "heuristic_fallback",
    }


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

async def _verify_single(claim: str, context: str, strictness: str | None = None) -> dict:
    """Verify a single claim. Attempt LLM judge first, fall back to heuristic."""

    # Try LLM judge
    llm_result = await _llm_judge(claim, context, strictness)
    if llm_result is not None:
        llm_result["method"] = "llm_judge"
        return llm_result

    # Fallback to heuristic
    result = _heuristic_score(claim, context, strictness)
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        api_keys_configured=len(API_KEYS),
        openrouter_configured=bool(OPENROUTER_API_KEY),
    )


@app.post(
    "/v1/verify",
    response_model=VerifyResponse,
    summary="Verify a single claim against its context",
)
async def verify_single(
    body: VerifyRequest,
    auth: dict = Depends(verify_auth),
):
    """Check whether an agent claim is supported by the provided context.

    Returns a hallucination score (0=supported to 1=hallucinated),
    a boolean verdict, an explanation, and relevant evidence from the context.
    """
    result = await _verify_single(body.claim, body.context, body.strictness)
    return VerifyResponse(**result)


@app.post(
    "/v1/verify-batch",
    response_model=BatchVerifyResponse,
    summary="Verify multiple claims in a single request",
)
async def verify_batch(
    body: BatchVerifyRequest,
    auth: dict = Depends(verify_auth),
):
    """Batch-verify up to 100 claims in one request. Each item is verified independently."""
    results = []
    for item in body.claims:
        result = await _verify_single(item.claim, item.context, item.strictness)
        results.append(VerifyResponse(**result))
    return BatchVerifyResponse(results=results)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.status_code},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=bool(os.environ.get("RELOAD", "0") == "1"),
        log_level="info",
    )
