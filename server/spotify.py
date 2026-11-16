import base64
import json
import os
import secrets
import time
import urllib.parse

import httpx
from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse

HERE = os.path.dirname(__file__)
SECRETS_FILE = os.path.join(HERE, "spotify_secrets.json")   # {"client_id": ..., "client_secret": ...}  (gitignored)
TOKEN_FILE = os.path.join(HERE, "spotify_token.json")       # written after you log in (gitignored)
REDIRECT_URI = "http://127.0.0.1:8000/spotify/callback"     # must match the Spotify dashboard exactly
SCOPES = "user-read-playback-state user-modify-playback-state user-read-currently-playing"
API = "https://api.spotify.com/v1"

router = APIRouter()
_state = None
_tokens = None


def _secrets():
    try:
        with open(SECRETS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def _load_tokens():
    global _tokens
    if _tokens is None:
        try:
            with open(TOKEN_FILE) as f:
                _tokens = json.load(f)
        except (FileNotFoundError, ValueError):
            _tokens = None
    return _tokens


def _save_tokens(t):
    global _tokens
    _tokens = t
    with open(TOKEN_FILE, "w") as f:
        json.dump(t, f)


def _basic_auth(s):
    raw = f"{s['client_id']}:{s['client_secret']}".encode()
    return "Basic " + base64.b64encode(raw).decode()


@router.get("/spotify/login")
async def login():
    global _state
    s = _secrets()
    if not s:
        return HTMLResponse("Missing server/spotify_secrets.json", status_code=500)
    _state = secrets.token_urlsafe(16)
    q = urllib.parse.urlencode({
        "client_id": s["client_id"], "response_type": "code",
        "redirect_uri": REDIRECT_URI, "scope": SCOPES, "state": _state,
    })
    return RedirectResponse("https://accounts.spotify.com/authorize?" + q)


@router.get("/spotify/callback")
async def callback(code: str = None, state: str = None, error: str = None):
    if error or not code or state != _state:
        return HTMLResponse(f"Spotify login failed: {error or 'bad state'}", status_code=400)
    s = _secrets()
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
            headers={"Authorization": _basic_auth(s)},
        )
    if r.status_code != 200:
        return HTMLResponse(f"Token exchange failed: {r.text}", status_code=400)
    t = r.json()
    _save_tokens({
        "access_token": t["access_token"],
        "refresh_token": t["refresh_token"],
        "expires_at": time.time() + t["expires_in"] - 60,
    })
    return HTMLResponse("<p>Spotify connected. You can close this tab.</p><script>window.close()</script>")


async def _access_token(client):
    t = _load_tokens()
    if not t:
        return None
    if time.time() >= t["expires_at"]:   # tokens last an hour, refresh quietly
        r = await client.post(
            "https://accounts.spotify.com/api/token",
            data={"grant_type": "refresh_token", "refresh_token": t["refresh_token"]},
            headers={"Authorization": _basic_auth(_secrets())},
        )
        if r.status_code != 200:
            print("spotify refresh failed:", r.text[:200])
            return None
        n = r.json()
        _save_tokens({
            "access_token": n["access_token"],
            "refresh_token": n.get("refresh_token", t["refresh_token"]),
            "expires_at": time.time() + n["expires_in"] - 60,
        })
    return _tokens["access_token"]


async def now_playing():
    """A small dict for the widget."""
    async with httpx.AsyncClient(timeout=10) as c:
        token = await _access_token(c)
        if not token:
            return {"connected": False}
        r = await c.get(f"{API}/me/player", headers={"Authorization": f"Bearer {token}"})
    if r.status_code == 204 or not r.content:   # nothing playing on any device
        return {"connected": True, "playing": False, "track": None}
    if r.status_code != 200:
        print("spotify now playing:", r.status_code, r.text[:200])
        return {"connected": True, "playing": False, "track": None}
    d = r.json()
    item = d.get("item") or {}
    images = (item.get("album") or {}).get("images") or []
    return {
        "connected": True,
        "playing": d.get("is_playing", False),
        "track": item.get("name"),
        "artist": ", ".join(a["name"] for a in item.get("artists", [])),
        "art": images[0]["url"] if images else None,
        "progress": d.get("progress_ms") or 0,
        "duration": item.get("duration_ms") or 0,
    }


ACTIONS = {
    "play": ("PUT", "/me/player/play"),
    "pause": ("PUT", "/me/player/pause"),
    "next": ("POST", "/me/player/next"),
    "previous": ("POST", "/me/player/previous"),
}


async def control(action):
    if action not in ACTIONS:
        return
    method, path = ACTIONS[action]
    async with httpx.AsyncClient(timeout=10) as c:
        token = await _access_token(c)
        if not token:
            return
        r = await c.request(method, API + path, headers={"Authorization": f"Bearer {token}"})
    if r.status_code >= 300:
        print("spotify", action, r.status_code, r.text[:200])