import asyncio
import json
from fastapi import FastAPI, WebSocket
from .physics import PhysicsWorld, WORLD_WIDTH, WORLD_HEIGHT
import httpx
import os
import re

app = FastAPI()
world = PhysicsWorld()
clients = set()

box_counter = 0

DT = 1/60

DEFAULT_WEATHER_LOCATION = {"label": "New Brunswick, NJ" , "lat": 40.4862, "lon": -74.4518}

# ---------- theme (colors) ----------
THEME_FILE = os.path.join(os.path.dirname(__file__), "theme_default.json")
DEFAULT_THEME = {"bg": "#000000", "body": "#00E5FF", "button": "#00E5FF"}
HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

def clean_theme(d):
    """Keep only known keys whose values are real #RRGGBB colors."""
    return {k: v for k, v in d.items() if k in DEFAULT_THEME and HEX.match(str(v))}

def load_theme():
    try:
        with open(THEME_FILE) as f:
            return {**DEFAULT_THEME, **clean_theme(json.load(f))}
    except (FileNotFoundError, ValueError):
        return dict(DEFAULT_THEME)

theme = load_theme()

async def broadcast_theme(skip=None):
    msg = json.dumps({"type": "theme", **theme})
    for ws in list(clients):
        if ws is skip:
            continue
        try:
            await ws.send_text(msg)
        except Exception:
            pass

@app.post("/theme")            # change the colors (used by Milo)
async def set_theme(new_theme: dict):
    theme.update(clean_theme(new_theme))
    await broadcast_theme()
    return theme

@app.post("/theme/save")       # make the current colors the default
async def save_theme():
    with open(THEME_FILE, "w") as f:
        json.dump(theme, f)
    return theme

@app.get("/status")            # is anyone looking at the dashboard right now?
async def status():
    return {"pages_open": len(clients), "theme": theme}


background_tasks = set()   # keeps running chat replies from being garbage-collected

async def handle_chat(obj_id, user_text):
    world.append_chat_message(obj_id, "user", user_text)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": user_text, "stream": False},
                timeout=60.0
            )
        reply = response.json()["response"]
    except Exception as e:
        print(f"ollama error: {e}")
        reply = "(M1L0 couldn't be reached. Is Ollama running?)"
    world.append_chat_message(obj_id, "assistant", reply)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "theme", **theme}))
        while True:
            msg = await websocket.receive_text()
            try:
                await handle_message(websocket, json.loads(msg))
            except Exception as e:
                print(f"bad message {msg[:80]}: {e}")   # skip it, keep the connection
    except Exception as e:
        print(f"websocket closed: {e}")
    finally:
        clients.discard(websocket)


async def handle_message(websocket, data):
    global box_counter
    kind = data["type"]

    if kind == "hit_test":
        print(f"hit_test at ({data['x']}, {data['y']})")
        found_body = world.get_body_at(data["x"], data["y"])
        await websocket.send_text(json.dumps({
            "type": "hit_test_result",
            "obj_id": found_body,          # None if nothing was there
        }))

    elif kind == "drag":
        world.bring_to_front(data["obj_id"])
        world.set_position(data["obj_id"], data["x"], data["y"])

    elif kind == "resize":
        found_body = world.get_body_at(data["x"], data["y"])
        if found_body is not None:
            world.resize_box(found_body, data["factor"])

    elif kind == "spawn":
        box_counter += 1
        new_id = f"box{box_counter}"
        chosen_type = data.get("window_type", "box")
        x, y = WORLD_WIDTH / 2, WORLD_HEIGHT / 3
        if chosen_type == "note":
            world.add_box(new_id, x, y, window_type="note", content="", width=160, height=220)
        elif chosen_type == "chat":
            world.add_box(new_id, x, y, window_type="chat", content=[], width=190, height=260)
        elif chosen_type == "weather":
            world.add_box(new_id, x, y, window_type="weather", content=dict(DEFAULT_WEATHER_LOCATION), width=270, height=150)
        else:
            world.add_box(new_id, x, y, window_type=chosen_type, width=120, height=120)

    elif kind == "remove":
        world.remove_box(data["obj_id"])

    elif kind == "resize_by_id":
        world.resize_box(data["obj_id"], data["factor"])

    elif kind == "set_size":
        world.set_size(data["obj_id"], float(data["w"]), float(data["h"]))

    elif kind == "set_location":
        obj_id = data["obj_id"]
        if world.types.get(obj_id) == "weather":
            world.set_content(obj_id, {
                "label": str(data.get("label", "Your Location"))[:40],
                "lat": float(data["lat"]),
                "lon": float(data["lon"]),
            })

    elif kind == "theme":
        theme.update(clean_theme(data))
        await broadcast_theme(skip=websocket)

    elif kind == "chat_message":
        task = asyncio.create_task(handle_chat(data["obj_id"], data["text"]))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

async def physics_loop():
    
    while True:
        world.step(DT)
        state = json.dumps({
            "type": "state",
            "objects": world.get_state()

        })
        for ws in list(clients):
            try:
                await ws.send_text(state)
            
            except Exception as e:
                print(f"error: {e}")
                pass

        await asyncio.sleep(DT)

@app.on_event("startup")
async def startup():
    asyncio.create_task(physics_loop())