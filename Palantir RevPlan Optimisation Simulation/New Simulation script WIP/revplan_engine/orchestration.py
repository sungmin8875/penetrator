"""
Publish an Orchestration Engine (OE) completion event at the end of a RevPlan
simulation run, so the OE's `finish-simulation` resume step advances.

The `/machine-learning/api/executions` trigger is fire-and-forget, so THIS event
is what actually closes the orchestration loop.

Ported VERBATIM (endpoint + CloudEvents `ce-*` headers + OAuth client-credentials)
from the LG PO-readout pipeline's `orchestration.py`. The correct mechanism is the
OE's native message API — NOT the opaque `.../ems-automation/.../hook/...` webhook
the notebook used before:

    POST {base_url}/oe/api/packages/{OE_PACKAGE_KEY}/messages
    headers:
        Authorization : Bearer <OAuth token, scope=orchestration-engine>
        ce-instanceid : <dpInstanceId>     # which OE instance to resume
        ce-type       : <event_type>       # which resume step (finish-simulation)
        ce-executionid: <unique>
        Content-Type  : application/json
    body: JSON (may be minimal / empty — finish-simulation carries no form)

The OE correlates on the `ce-instanceid` + `ce-type` HEADERS (not the body), which
is why the old approach — dpInstanceId in the JSON body, no ce-type, wrong URL —
returned 2xx yet never resumed anything.

Config (set in the MLWB environment / .env — see README):
    OE_PACKAGE_KEY     the OE package that owns the simulation process
    OE_EVENT_TYPE      defaults to 'finish-simulation'
    oe_client_id       OAuth client with the orchestration-engine scope
    oe_client_secret
    CELONIS_BASE_URL   team URL (falls back to CELONIS_URL, then the LG default)
"""

from __future__ import annotations

import json
import os
import time

import requests

# Event name the simulation OE's resume step listens for (the `ce-type`).
DEFAULT_SIMULATION_EVENT = "finish-simulation"

# Scope for the Orchestration Engine token.
_OE_SCOPE = "orchestration-engine"

# Cache so we mint one OE token per session.
_OE_ACCESS_TOKEN: "str | None" = None


def base_url() -> str:
    """Team URL (no trailing slash). Reads CELONIS_BASE_URL / CELONIS_URL."""
    u = os.environ.get("CELONIS_BASE_URL") or os.environ.get(
        "CELONIS_URL", "https://lg-innotek.eu-1.celonis.cloud"
    )
    if not u.startswith(("http://", "https://")):
        u = "https://" + u
    return u.rstrip("/")


def get_oe_access_token(force_refresh: bool = False) -> str:
    """
    Mint (and cache) an Orchestration Engine access token via the OAuth2
    client-credentials flow (oauthlib BackendApplicationClient + requests_oauthlib
    OAuth2Session + HTTP Basic auth + scope=['orchestration-engine']). Reads the
    dedicated oe_client_id / oe_client_secret (the OE client with the
    orchestration-engine scope; see README).

    Imported lazily so `import revplan_engine.orchestration` works without oauthlib
    installed; the notebook's Dependencies cell installs it for the run.
    """
    global _OE_ACCESS_TOKEN
    if _OE_ACCESS_TOKEN and not force_refresh:
        return _OE_ACCESS_TOKEN

    from oauthlib.oauth2 import BackendApplicationClient
    from requests_oauthlib import OAuth2Session
    from requests.auth import HTTPBasicAuth

    client_id     = os.getenv("oe_client_id")     or os.getenv("OE_CLIENT_ID")
    client_secret = os.getenv("oe_client_secret") or os.getenv("OE_CLIENT_SECRET")
    if not client_id or not client_secret or client_id.startswith("<"):
        raise RuntimeError(
            "OE OAuth credentials not configured. Set oe_client_id / "
            "oe_client_secret (the OE client with the orchestration-engine scope) "
            "in the environment or .env (see README)."
        )

    auth   = HTTPBasicAuth(client_id, client_secret)
    client = BackendApplicationClient(client_id=client_id)
    oauth  = OAuth2Session(client=client)
    token  = oauth.fetch_token(
        token_url=f"{base_url()}/oauth2/token",
        auth=auth,
        scope=[_OE_SCOPE],
    )
    _OE_ACCESS_TOKEN = token["access_token"]
    return _OE_ACCESS_TOKEN


# Simulation OE package KEY (the package whose `finish-simulation` step we resume).
# This is the package *key* (slug, underscores) — NOT the package *id* (the hyphenated
# UUID in the Studio URL after /packages/). The /oe/api/packages/{key}/messages endpoint
# wants the key. Overridable via the OE_PACKAGE_KEY env var.
DEFAULT_OE_PACKAGE_KEY = "36f39662_626e_41c2_be62_fcd0ead2b2ce"


def _oe_package_key(explicit: str = "") -> str:
    """OE package key, from the explicit arg or the OE_PACKAGE_KEY env var."""
    key = explicit or os.environ.get("OE_PACKAGE_KEY", DEFAULT_OE_PACKAGE_KEY)
    if not key or key.startswith("<"):
        raise RuntimeError(
            "OE package key not configured. Pass package_key=... or set "
            "OE_PACKAGE_KEY in the environment / .env (see README)."
        )
    return key


def oe_completion_event(dp_instance_id: str, event_type: str, package_key: str = "",
                        body: "dict | None" = None, execution_id=None,
                        timeout: int = 60) -> requests.Response:
    """
    POST a completion event (CloudEvents message) to the OE package.

    Args:
        dp_instance_id: instance id of the OE execution to complete (-> ce-instanceid).
        event_type:     the event name the OE listens for (-> ce-type).
        package_key:    OE package key (falls back to OE_PACKAGE_KEY env var).
        body:           JSON-serialisable event context (optional for finish-simulation).
        execution_id:   unique id for this event; defaults to a nanosecond clock.

    Returns the requests.Response — inspect .ok / .status_code for the result.
    """
    package_key = _oe_package_key(package_key)
    if execution_id is None:
        execution_id = time.time_ns()
    url = f"{base_url()}/oe/api/packages/{package_key}/messages"
    headers = {
        "Authorization":  f"Bearer {get_oe_access_token()}",
        "ce-instanceid":  str(dp_instance_id),
        "ce-type":        str(event_type),
        "ce-executionid": str(execution_id),
        "Content-Type":   "application/json",
    }
    data = json.dumps(body or {}).encode("utf-8")
    return requests.post(url, headers=headers, data=data, timeout=timeout)


def emit_finished_simulation(dp_instance_id: str, body: "dict | None" = None,
                             package_key: str = "",
                             event_type: str = DEFAULT_SIMULATION_EVENT,
                             execution_id=None,
                             raise_on_error: bool = True) -> "requests.Response | None":
    """
    Emit the `finish-simulation` completion event for this run's OE instance.

    Returns the Response, or None when skipped (no dpInstanceId, or no package key
    configured — e.g. a manual run without an Orchestration Engine instance).

    Unlike the PO reference (which only logs), this RAISES on a non-2xx by default:
    a failed resume strands the OE, so it must be loud rather than silently swallowed.
    """
    if not dp_instance_id:
        print("[oe] No dpInstanceId — skipping completion event "
              "(manual run without an Orchestration Engine instance).")
        return None
    try:
        key = _oe_package_key(package_key)
    except RuntimeError as e:
        print(f"[oe] {e}\n[oe] Skipping completion event.")
        return None

    body = body or {}
    body_bytes = json.dumps(body).encode("utf-8")
    print(f"[oe] → POST {base_url()}/oe/api/packages/{key}/messages")
    print(f"[oe]   ce-instanceid={dp_instance_id}  ce-type={event_type}")
    print(f"[oe]   payload ({len(body_bytes)} bytes)")

    resp = oe_completion_event(dp_instance_id, event_type, key,
                               body=body, execution_id=execution_id)
    if resp.ok:
        print(f"[oe] Sent '{event_type}' completion event for instance "
              f"{dp_instance_id} (HTTP {resp.status_code}). "
              f"Response: {resp.text[:300] or '<empty>'}")
    else:
        print(f"[oe][ERROR] '{event_type}' event failed: HTTP "
              f"{resp.status_code} — {resp.text[:300]}")
        if raise_on_error:
            resp.raise_for_status()
    return resp
