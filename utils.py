"""
Infrastructure utilities: DB pool, config, LLM inference, JSON parsing.

Agent grounding (phase detection, navigation, action substitution) lives in
core.agent_grounding and is re-exported below for backward compatibility.
"""
import json
import re
import ast
import os
import configparser
import logging

try:
    import psycopg2
    from psycopg2 import pool, extras
except ImportError:  # sanity / offline runs without DB
    psycopg2 = None
    pool = None
    extras = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    def cosine_similarity(a, b):
        import numpy as np
        a = np.asarray(a)
        b = np.asarray(b)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        denom = (np.linalg.norm(a, axis=1, keepdims=True) * np.linalg.norm(b, axis=1, keepdims=True).T)
        denom = np.where(denom == 0, 1.0, denom)
        return (a @ b.T) / denom

try:
    from pydantic import BaseModel, ValidationError
except ImportError:  # offline sanity runs
    class ValidationError(Exception):
        pass

    class BaseModel:  # minimal stub for json validation paths
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        def dict(self):
            return self.__dict__

        @classmethod
        def model_validate(cls, obj):
            return cls(**obj) if isinstance(obj, dict) else cls()
try:
    from pydub import AudioSegment
    from pydub.playback import play
except ImportError:  # 平台离线镜像可能未装 pydub；主流程不依赖音频播放
    AudioSegment = None
    play = None
import inspect
import random
import numpy as np

try:
    from transformers import pipeline
except ImportError:
    pipeline = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

_LOCAL_TEXT_GENERATORS = {}
_LOCAL_EMBED_MODELS = {}
_MISSING = object()

# def get_connection_pool(config):
#     """
#     Creates and returns a database connection pool using psycopg2 based on provided configuration details.

#     Parameters:
#     - config (ConfigParser): An object containing database configuration details. Expected to have 'DB' section with 'DB_NAME', 'DB_USER', and 'DB_PASS' keys.

#     Returns:
#     - psycopg2.pool.SimpleConnectionPool: A connection pool object with a minimum of 1 connection and a maximum of 25 connections.

#     Example usage:
#     config = configparser.ConfigParser()
#     config.read('config.ini')
#     pool = get_connection_pool(config)
#     """
#     # Create a DB connection pool based on config details
#     connection_pool = pool.SimpleConnectionPool(
#         1,  # minconn
#         400,  # maxconn
#         dbname=config.get('DB', 'DB_NAME'),
#         user=config.get('DB', 'DB_USER'),
#         password=config.get('DB', 'DB_PASS')
#     )

#     return connection_pool

def get_connection_pool(config):
    """
    Build a psycopg2 SimpleConnectionPool from ``config.ini`` ``[DB]`` section.
    Fails fast with a clear error (no silent ``None``) so experiments are reproducible
    once the same ini + Postgres are available.
    """
    if config is None:
        raise ValueError(
            "get_connection_pool: config is None. Use load_config with an absolute path to config.ini."
        )
    if not config.has_section("DB"):
        raise ValueError("config.ini must define a [DB] section with DB_NAME, DB_USER, DB_PASS, DB_HOST, DB_PORT.")

    db_name = config.get("DB", "DB_NAME")
    db_user = config.get("DB", "DB_USER")
    db_pass = config.get("DB", "DB_PASS", fallback="")
    db_host = config.get("DB", "DB_HOST", fallback="localhost")
    db_port = config.get("DB", "DB_PORT", fallback="5432")
    maxconn = config.getint("DB", "DB_MAXCONN", fallback=5)

    try:
        return pool.SimpleConnectionPool(
            minconn=1,
            maxconn=max(1, maxconn),
            dbname=db_name,
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
        )
    except Exception as e:
        raise RuntimeError(
            f"PostgreSQL connection pool failed ({db_host}:{db_port}/{db_name} as {db_user}). "
            "Check config/config.ini [DB] and that the server is running."
        ) from e

def count_connections(pool):
    try:
        return {
            'minconn': pool.minconn,
            'maxconn': pool.maxconn,
            'usedconn': pool._used,
            'freeconn': pool._pool
        }
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def load_config(config_file):
    """
    Loads the config file from configuration file (ie: config.ini). See config.ini.example for details.
    :param config_file: path and filename for config.ini file
    :return: configparser.ConfigParser object or None if the file does not exist
    """
    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    # Checking if the config file was empty or improperly formatted
    if not config.sections():
        print(f"Config file {config_file} is empty or improperly formatted.")
        return None

    return config

def setup_logger(config):
    """
    Setup logger for system.
    :param config: config object for use in setting up installation specific configuration variables.
    :return: logger object
    """
    # setup logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=config.get('DEFAULT', 'LOG_FILE'), format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    return logger

def estimate_token_usage(text):
    """
    Estimate the number of tokens in a given text using basic assumptions.
    
    Parameters:
    text (str): The text for which to estimate token usage.
    
    Returns:
    int: The estimated number of tokens.
    """
    # Remove any non-word characters to approximate token count
    words = re.findall(r'\w+', text)
    
    # Approximate number of tokens (considering punctuation, special characters as separate tokens)
    # Generally, average token length in GPT models is around 4 characters
    # Adding a factor to account for special tokens and punctuation
    estimated_tokens = sum(len(word) // 4 + 1 for word in words)
    
    return estimated_tokens

def _get_local_chat_model(config, model):
    configured_default = config.get(
        'DEFAULT',
        'LOCAL_CHAT_MODEL',
        # fallback='/mnt/nfsData19/Zhaoshuyuan/Houxinrui/model/Qwen3.5-9B',
        fallback='/data/ZhaoShuyuan/Zhaoshuyuan/HouXinrui/Baseline/Qwen3.5-9B',
    )
    alias_map = {
        'gpt-4o': configured_default,
        'gpt-4': configured_default,
        'gpt-4-turbo': configured_default,
        'gpt-4o-mini': configured_default,
        'gpt-3.5': configured_default,
        'gpt-3.5-turbo': configured_default,
        'meta-llama-3-70b-instruct': configured_default,
    }
    return alias_map.get(model, model if model else configured_default)

def _get_local_embed_model(config, model):
    configured_default = config.get('DEFAULT', 'LOCAL_EMBED_MODEL', fallback='sentence-transformers/all-MiniLM-L6-v2')
    return model if model else configured_default

def _is_qwen3_model_ref(model_ref: str) -> bool:
    return "qwen3" in (model_ref or "").lower()


def _apply_chat_template(tok, messages, **kwargs):
    """Qwen3 chat templates accept enable_thinking=False for non-reasoning mode."""
    extra = dict(kwargs)
    path = (getattr(tok, "name_or_path", None) or "").lower()
    if _is_qwen3_model_ref(path):
        extra.setdefault("enable_thinking", False)
    try:
        return tok.apply_chat_template(messages, **extra)
    except TypeError:
        extra.pop("enable_thinking", None)
        return tok.apply_chat_template(messages, **extra)


def _flash_attn_installed() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def _attn_implementation_candidates():
    """Prefer FA2 when the wheel exists; otherwise PyTorch SDPA (no extra package)."""
    impls = []
    if _flash_attn_installed():
        impls.append("flash_attention_2")
    impls.append("sdpa")
    impls.append(None)
    return impls


def _generation_load_kw(resolved_model, attn_implementation=None):
    import torch

    load_kw = {"device_map": "auto", "trust_remote_code": True}
    if _is_qwen3_model_ref(resolved_model) or attn_implementation in (
        "flash_attention_2",
        "sdpa",
    ):
        load_kw["torch_dtype"] = torch.bfloat16
    if attn_implementation:
        load_kw["attn_implementation"] = attn_implementation
    return load_kw


def _build_text_generation_pipeline(resolved_model, load_kw):
    return pipeline(
        "text-generation",
        model=resolved_model,
        tokenizer=resolved_model,
        **load_kw,
    )


def _resolved_attn_implementation(pipe) -> str:
    cfg = getattr(getattr(pipe, "model", None), "config", None)
    impl = getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation", None)
    return str(impl or "default")


def _empty_cuda_cache():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _get_text_generator(config, model_name, logger=None):
    resolved_model = _get_local_chat_model(config, model_name)
    if resolved_model not in _LOCAL_TEXT_GENERATORS:
        log = logger or logging.getLogger(__name__)
        if not _flash_attn_installed():
            log.info(
                "flash-attn not installed; loading %s with PyTorch SDPA "
                "(do not compile flash-attn in the job)",
                resolved_model,
            )
        pipe = None
        last_exc = None
        for impl in _attn_implementation_candidates():
            try:
                pipe = _build_text_generation_pipeline(
                    resolved_model,
                    _generation_load_kw(resolved_model, attn_implementation=impl),
                )
                break
            except (ImportError, ValueError, TypeError, OSError, RuntimeError) as exc:
                last_exc = exc
                log.warning(
                    "attn_implementation=%s failed for %s (%s: %s); trying next",
                    impl or "default",
                    resolved_model,
                    type(exc).__name__,
                    exc,
                )
                _empty_cuda_cache()
        if pipe is None:
            raise last_exc or RuntimeError(
                f"Failed to load chat model {resolved_model}"
            )
        # Some checkpoints ship a tiny default max_length (e.g. 20) on generation_config; it
        # interacts badly with max_new_tokens and can yield empty generations + JSON errors.
        model_obj = getattr(pipe, "model", None)
        gc = getattr(model_obj, "generation_config", None) if model_obj is not None else None
        if gc is not None:
            ml = getattr(gc, "max_length", None)
            if ml is not None and ml < 512:
                try:
                    gc.max_length = None
                except (TypeError, ValueError, AttributeError):
                    try:
                        gc.max_length = 32768
                    except Exception:
                        pass
        impl_name = _resolved_attn_implementation(pipe)
        # Print as well as log: preload often uses a module logger with no
        # handlers, so INFO never reaches result.log (stdout is redirected there).
        print(
            f"[model] Loaded chat model {resolved_model} attn_implementation={impl_name}",
            flush=True,
        )
        log.info(
            "Loaded chat model %s attn_implementation=%s",
            resolved_model,
            impl_name,
        )
        _LOCAL_TEXT_GENERATORS[resolved_model] = pipe
    return _LOCAL_TEXT_GENERATORS[resolved_model], resolved_model


def preload_local_chat_models(config, *model_paths, logger=None):
    """Load chat models into the process cache before the first inference call."""
    seen: set[str] = set()
    for path in model_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        if logger:
            logger.info("Preloading chat model: %s", path)
        _get_text_generator(config, path, logger=logger)


def _get_embedder(config, model_name):
    resolved_model = _get_local_embed_model(config, model_name)
    if resolved_model not in _LOCAL_EMBED_MODELS:
        _LOCAL_EMBED_MODELS[resolved_model] = SentenceTransformer(resolved_model)
    return _LOCAL_EMBED_MODELS[resolved_model]


def _local_llm_generate(pipe, user_prompt, max_new_tokens, temperature=0.0, json_mode=False):
    """
    Generate with the local chat model. For JSON, use chat template + assistant prefix '{'
    so Qwen/Instruct models do not reply with markdown analysis.
    """
    import torch

    model = pipe.model
    tok = pipe.tokenizer
    device = model.device
    pad_id = tok.eos_token_id
    if getattr(tok, "pad_token_id", None) is None and pad_id is not None:
        tok.pad_token_id = pad_id

    json_prefix = ""
    has_chat = bool(getattr(tok, "chat_template", None)) and hasattr(tok, "apply_chat_template")

    if json_mode and has_chat:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON-only API. Output exactly one valid JSON object. "
                    "No markdown, no code fences, no explanation, no analysis. "
                    "The first character of your reply must be {. "
                    "Keep every string value concise (under 60 words)."
                ),
            },
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "{"},
        ]
        try:
            text = _apply_chat_template(
                tok,
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
            json_prefix = "{"
        except TypeError:
            messages = messages[:2]
            text = _apply_chat_template(
                tok, messages, tokenize=False, add_generation_prompt=True,
            )
            text = text + "{"
            json_prefix = "{"
    elif json_mode:
        text = (
            user_prompt
            + "\n\nReply with exactly one JSON object only. "
            "First character must be {. No other text.\n\n{"
        )
        json_prefix = "{"
    elif has_chat:
        text = _apply_chat_template(
            tok,
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = user_prompt

    model_max = getattr(tok, "model_max_length", None) or 8192
    if model_max > 100000 or model_max < 256:
        model_max = 8192
    max_input = min(model_max - max_new_tokens - 16, 8192)
    max_input = max(max_input, 256)

    inputs = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    gen_kw = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": pad_id,
        "eos_token_id": pad_id,
    }
    if temperature and temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = float(temperature)
    else:
        gen_kw["do_sample"] = False

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kw)

    new_ids = output_ids[0, input_len:]
    generated = tok.decode(new_ids, skip_special_tokens=True).strip()
    if json_prefix and not generated.startswith("{"):
        generated = json_prefix + generated
    return generated


def openai_speak(config, text):
    raise NotImplementedError("openai_speak has been disabled. This project is configured for local-only models.")

def _sanitize_llm_json_text(text: str) -> str:
    """Strip chat-role prefixes and template leaks before JSON extraction."""
    t = (text or "").strip().lstrip("\ufeff")
    t = re.sub(r"^\{?\s*(user|assistant|system)\s*\n+", "", t, flags=re.IGNORECASE)
    # Qwen sometimes echoes instruction fragments like "{_completion` and values..."
    t = re.sub(r"^\{[_a-z][^\"{]*", "", t, flags=re.IGNORECASE)
    while t and not t.startswith("{") and "{" in t:
        idx = t.find("{")
        if _json_object_starts_at(t, idx):
            t = t[idx:].lstrip()
            break
        t = t[idx + 1 :].lstrip()
    # Model echoed JSON schema instructions instead of an object.
    if t.startswith('"') and "state" in t[:120] and not t.startswith('{"'):
        for marker in ('{"query"', '{"state"', "{"):
            pos = t.find(marker)
            if pos >= 0 and _json_object_starts_at(t, pos):
                t = t[pos:].lstrip()
                break
    return t


def _json_object_starts_at(text: str, start: int) -> bool:
    """Reject prose like '{a different action...' that is not a JSON object."""
    if start < 0 or start >= len(text) or text[start] != "{":
        return False
    j = start + 1
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text):
        return False
    return text[j] in ('"', "}", "[") or text[j].isdigit()


def _extract_first_json_object(text: str) -> str | None:
    """
    Extract the first JSON object substring from a model output.
    We do NOT call any LLM for "fixing" JSON. If we can't parse, we fail fast
    and let the caller retry generation.
    """
    text = _sanitize_llm_json_text(text)
    if not text:
        return None

    # Prefer fenced blocks.
    m = re.search(r"```json\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
    if m and "{" in m.group(1) and "}" in m.group(1):
        return m.group(1).strip()

    # Try each '{' start (handles '{user\n{ ...' garbage prefixes).
    search_from = 0
    while search_from < len(text):
        start = text.find("{", search_from)
        if start == -1:
            return None
        if not _json_object_starts_at(text, start):
            search_from = start + 1
            continue
        depth = 0
        in_str = False
        esc = False
        found = None
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        found = text[start : i + 1].strip()
                        break
        if found:
            try:
                json.loads(found)
                return found
            except Exception:
                try:
                    json.loads(_normalize_jsonish(found))
                    return _normalize_jsonish(found)
                except Exception:
                    pass
        search_from = start + 1
    return None


def _normalize_jsonish(text: str) -> str:
    """
    Heuristics to turn common LLM 'almost-JSON' into valid JSON without calling another LLM.
    - Quote numeric keys: { 0: "LOC" } -> { "0": "LOC" }
    - Quote bareword keys: { foo: "bar" } -> { "foo": "bar" }
    - Remove trailing commas before } or ].
    """
    t = (text or "").strip()
    # remove trailing commas
    t = re.sub(r",\s*([\}\]])", r"\1", t)
    # quote numeric keys
    t = re.sub(r"([\{\[,]\s*)(\d+)\s*:", r'\1"\2":', t)
    # quote bareword keys (avoid already-quoted keys)
    t = re.sub(r"([\{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:", r'\1"\2":', t)
    return t


def _placeholder_actor_next_state() -> dict:
    return {"observation": "pending", "inventory": "", "valid_receptacles": []}


def normalize_state_with_query_dict(res: dict) -> dict:
    """Coerce flat observation/inventory JSON into StateWithQuery shape."""
    if not isinstance(res, dict):
        return res
    lower = {str(k).lower(): v for k, v in res.items()}
    if "state" in lower and ("query" in lower or "task_completion" in lower):
        return lower
    if "observation" in lower and (
        "inventory" in lower or "valid_receptacles" in lower
    ):
        state = {
            k: lower[k]
            for k in ("observation", "inventory", "valid_receptacles")
            if k in lower
        }
        return {
            "query": lower.get("query"),
            "state": state,
            "task_completion": lower.get("task_completion", False),
        }
    return lower


def normalize_actor_dict(res: dict) -> dict:
    """Ensure Actor JSON has valid next_states entries for pydantic validation."""
    if not isinstance(res, dict) or "actions" not in res:
        return res
    actions = list(res.get("actions") or [])
    n = len(actions)
    if n == 0:
        return res
    responses = list(res.get("responses") or [])
    next_states = list(res.get("next_states") or [])
    while len(responses) < n:
        responses.append("")
    fixed: list[dict] = []
    for i in range(n):
        ns = next_states[i] if i < len(next_states) else {}
        if isinstance(ns, dict) and ns.get("observation") is not None:
            fixed.append(ns)
        else:
            fixed.append(_placeholder_actor_next_state())
    res["actions"] = actions
    res["responses"] = responses[:n]
    res["next_states"] = fixed
    return res


def _extract_json_field(text: str, field: str):
    """Extract a JSON string or null field from possibly truncated output."""
    m = re.search(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', text)
    if m:
        return m.group(1).encode().decode("unicode_escape", errors="replace")
    if re.search(rf'"{re.escape(field)}"\s*:\s*null\b', text):
        return None
    return _MISSING


def _extract_partial_action_prediction_json(raw: str) -> dict | None:
    """Recover ActionPrediction when model JSON is truncated mid-object."""
    text = str(raw or "")
    if '"predicted_action"' not in text and '"query"' not in text:
        return None
    predicted_action = _extract_json_field(text, "predicted_action")
    query = _extract_json_field(text, "query")
    if predicted_action is _MISSING and query is _MISSING:
        return None
    predicted_response = _extract_json_field(text, "predicted_response")
    reasoning = _extract_json_field(text, "reasoning")
    return {
        "query": None if query is _MISSING else query,
        "predicted_action": None if predicted_action is _MISSING else predicted_action,
        "predicted_response": "" if predicted_response is _MISSING else (predicted_response or ""),
        "reasoning": "" if reasoning is _MISSING else (reasoning or ""),
    }


def _extract_partial_actor_json(raw: str) -> dict | None:
    """Recover first Actor action when model JSON is truncated mid-object."""
    text = str(raw or "")
    if '"actions"' not in text:
        return None
    m = re.search(r'"actions"\s*:\s*\[\s*"((?:\\.|[^"\\])*)"', text)
    if not m:
        return None
    action = m.group(1).encode().decode("unicode_escape", errors="replace")
    if not action.strip():
        return None
    return normalize_actor_dict(
        {
            "actions": [action],
            "responses": [""],
            "next_states": [_placeholder_actor_next_state()],
        }
    )


def json_parser(config, input, max_attemps=5):
    """
    Strict JSON parsing (no LLM repair). Attempts:
    - direct json.loads
    - extract first JSON object block from text, then json.loads
    """
    if input is None:
        raise ValueError("json_parser: input is None")

    raw = _sanitize_llm_json_text(str(input))
    if not raw:
        raise ValueError(
            "json_parser: empty input after strip — the model returned no text to parse as JSON. "
            "Check generation (OOM, max_new_tokens) or prompt length."
        )
    try:
        res = json.loads(raw)
    except Exception:
        partial_actor = _extract_partial_actor_json(raw)
        if partial_actor is not None:
            return partial_actor

        partial_action = _extract_partial_action_prediction_json(raw)
        if partial_action is not None:
            return partial_action

        extracted = _extract_first_json_object(raw)
        if extracted is None or not str(extracted).strip():
            raise ValueError(
                "json_parser: no JSON object found in model output. "
                f"Preview (first 500 chars): {raw[:500]!r}"
            ) from None
        extracted = str(extracted).strip()

        try:
            res = json.loads(extracted)
        except Exception:
            normalized = _normalize_jsonish(extracted)
            if not str(normalized).strip():
                raise ValueError(
                    "json_parser: normalization produced empty string. "
                    f"Original extracted (first 500 chars): {extracted[:500]!r}"
                ) from None
            try:
                res = json.loads(normalized)
            except Exception:
                try:
                    lit = ast.literal_eval(extracted)
                    res = lit
                except (ValueError, SyntaxError) as se:
                    raise ValueError(
                        "json_parser: could not parse as JSON. "
                        f"raw preview: {raw[:400]!r}; normalized preview: {str(normalized)[:400]!r}"
                    ) from se

    if isinstance(res, dict):
        res = {str(k).lower(): res[k] for k in res.keys()}
        if "actions" in res:
            res = normalize_actor_dict(res)
        elif "observation" in res and (
            "inventory" in res or "valid_receptacles" in res
        ):
            res = normalize_state_with_query_dict(res)
        return res
    return res

def get_gpt_response(config, gpt_prompt, model, temperature=0.3, max_token=512, response_type='text'):
    """
    Local-only replacement for legacy API-based generation.
    """
    pipe, resolved_model = _get_text_generator(config, model)
    json_mode = response_type == "json_object"
    if json_mode:
        json_max = config.getint('DEFAULT', 'LOCAL_JSON_MAX_TOKENS', fallback=2048)
        max_token = max(max_token, json_max)
    text = _local_llm_generate(
        pipe,
        gpt_prompt,
        max_new_tokens=max_token,
        temperature=temperature,
        json_mode=json_mode,
    )
    if not text:
        raise RuntimeError(
            f"Local model `{resolved_model}` returned empty output "
            f"(response_type={response_type!r}, max_new_tokens={max_token})."
        )
    return text

def get_octoml_response(config,llm_prompt,model='mixtral-8x7b-instruct', system_prompt="Below is an instruction that describes a task. Write a response that appropriately completes the request.", schema=None):
    combined_prompt = f"{system_prompt}\n\n{llm_prompt}"
    response_type = 'json_object' if schema else 'text'
    return get_gpt_response(config, combined_prompt, model=model, response_type=response_type)


def get_groq_response(config, llm_prompt, model='llama3-8b-8192', system_prompt=""):
    combined_prompt = f"{system_prompt}\n\n{llm_prompt}" if system_prompt else llm_prompt
    return get_gpt_response(config, combined_prompt, model=model, temperature=0)

def get_nvidia_res(config, prompt, model="meta/llama3-70b"):
    return get_gpt_response(config, prompt, model=model, temperature=0.5, max_token=1024)


def get_azure_response(config, gpt_prompt, model, temperature=0, max_token=2000, response_type='text'):
    return get_gpt_response(config, gpt_prompt, model=model, temperature=temperature, max_token=max_token, response_type=response_type)



def get_response(model, prompt, json=False, schema=None, config=None, count_token=False):
    if config is None:
        raise ValueError("config is required for local model inference.")

    res = get_gpt_response(config, prompt, model, temperature=0, response_type='json_object' if json else 'text')
    if json:
        res = json_parser(config, res)
    
    return res, {'sent': estimate_token_usage(prompt), 'received': estimate_token_usage(str(res))}

def get_embedding(config, text, model):
    text = text.replace("\n", " ")
    try:
        embedder = _get_embedder(config, model)
        embedding = embedder.encode(text, normalize_embeddings=True)
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
    except Exception as e:
        print(f"An error occurred while fetching local embeddings: {str(e)}")
        raise

def get_cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def retry(max_attempts, func, val_func, *args, **kwargs):
    attempt = 0
    token_count = {'sent': 0, 'received': 0}
    last_result = None
    last_error = None
    while attempt < max_attempts:
        result, new_token_count = func(*args, **kwargs)
        token_count['sent'] += new_token_count['sent']
        token_count['received'] += new_token_count['received']
        last_result = result
        try:
            if inspect.isclass(val_func) and issubclass(val_func, BaseModel):
                payload = result
                if isinstance(payload, dict):
                    name = getattr(val_func, "__name__", "")
                    if name == "StateWithQuery":
                        payload = normalize_state_with_query_dict(payload)
                validated_result = val_func(**payload)
                dump = getattr(validated_result, "model_dump", None)
                if callable(dump):
                    return dump(), token_count
                return validated_result.dict(), token_count
            else:
                if val_func(result):
                    return result, token_count
        except ValidationError as ve:
            last_error = ve
            print(result)
            print(f"Validation failed on attempt {attempt + 1}: {ve}")
        except Exception as e:
            last_error = e
            print(f"An error occurred on attempt {attempt + 1}: {e}")
            
        print(f"Validation failed on attempt {attempt + 1}: {result}")
        attempt += 1

    val_name = getattr(val_func, "__name__", val_func.__class__.__name__)
    result_preview = repr(last_result)
    if len(result_preview) > 2000:
        result_preview = result_preview[:2000] + "...(truncated)"
    err_preview = repr(last_error) if last_error is not None else "None"
    if len(err_preview) > 1200:
        err_preview = err_preview[:1200] + "...(truncated)"
    raise ValueError(
        "retry() exhausted attempts without valid output.\n"
        f"- validator: {val_name}\n"
        f"- attempts: {max_attempts}\n"
        f"- last_error: {err_preview}\n"
        f"- last_result: {result_preview}\n"
        f"- token_count: {token_count}\n"
    )

def load_variation(env, set, cutoff=False):
    variations = []
    set_key = str(set or "").strip().lower()
    orig_key = set_key
    # AlfWorld-friendly aliases → SciWorld-style getters on the adapter
    if set_key in ("seen", "eval_id", "eval_in_distribution", "valid_seen"):
        set_key = "dev"
    elif set_key in ("unseen", "eval_ood", "eval_out_of_distribution", "valid_unseen"):
        set_key = "test"
    if (set_key == "train"):
        variations = list(env.getVariationsTrain())
    elif (set_key == "test"):
        variations = list(env.getVariationsTest())
        if cutoff:
            test_len = min(50, len(variations))
            random.seed(1)
            random.shuffle(variations)
            variations = variations[:test_len]
    elif (set_key == "dev"):
        variations = list(env.getVariationsDev())
        # SciWorld mini-dev only: SET=dev → first 3. AlfWorld SET=seen/valid_seen keeps full split.
        if orig_key == "dev":
            variations = variations[:3]
    elif (set_key == "test_mini_2"):
        variations = list(env.getVariationsTest()) 
        variations = variations[3:10] 
    elif (set_key == "test_mini"):
        variations = list(env.getVariationsTest()) 
        variations = variations[:5] 
    elif (set_key == "test_mini_mini"):
        variations = list(env.getVariationsTest()) 
        variations = variations[:1] 
    else:
        raise ValueError("ERROR: Unknown set to evaluate on (" + str(set) + ")")
    return variations




# --- Agent grounding (phase, navigation, substitution) ---
import core.agent_grounding as _grounding
from core.agent_grounding import *  # noqa: F403, F401

# Star import skips leading-underscore helpers; re-export for `from utils import _foo`.
for _name in dir(_grounding):
    if _name.startswith("__"):
        continue
    globals().setdefault(_name, getattr(_grounding, _name))
