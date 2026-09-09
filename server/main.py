import asyncio
import json
from fastapi import FastAPI, WebSocket
from .physics import PhysicsWorld, WORLD_WIDTH, WORLD_HEIGHT
import httpx

app = FastAPI()
world = PhysicsWorld()
clients = set()

box_counter = 0

DT = 1/60

@app.websocket("/ws")

async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)

    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)
            if(data["type"] == "hit_test"):
                print(f"hit_test at ({data['x']}, {data['y']})")
                found_body = world.get_body_at(data["x"], data["y"])
                if found_body is not None:
                    await websocket.send_text(json.dumps({
                        "type": "hit_test_result",
                        "obj_id": found_body
                    }));
                else:
                    await websocket.send_text(json.dumps({
                        "type": "hit_test_result",
                        "obj_id": None
                    }));
            elif(data["type"] == "drag"):
                world.set_position(data["obj_id"], data["x"], data["y"])
            elif(data["type"] == "resize"):
                found_body = world.get_body_at(data["x"], data["y"])
                if found_body is not None:
                    world.resize_box(found_body, data["factor"])
            elif(data["type"] == "spawn"):
                global box_counter
                box_counter += 1
                new_id = f"box{box_counter}"
                chosen_type = data.get("window_type", "box")
                if chosen_type == "note":
                    world.add_box(new_id, WORLD_WIDTH / 2, WORLD_HEIGHT / 2, window_type="note", content="", width=350, height=250)
                elif chosen_type == "chat":
                    world.add_box(new_id, WORLD_WIDTH / 2, WORLD_HEIGHT / 2, window_type="chat", content=[], width=350, height=400)
                else:
                    world.add_box(new_id, WORLD_WIDTH / 2, WORLD_HEIGHT / 2, window_type=chosen_type)
            elif(data["type"] == "remove"):
                world.remove_box(data["obj_id"])
            elif(data["type"] == "resize_by_id"):
                world.resize_box(data["obj_id"], data["factor"])
            elif(data["type"] == "chat_message"):
                obj_id = data["obj_id"]
                user_text = data["text"]

                #placeholder for the claude API#

                world.append_chat_message(obj_id, "user" , user_text)

                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://localhost:11434/api/generate",
                        json={"model": "llama3.2", "prompt": user_text, "stream": False},
                        timeout=60.0
                    )
                reply = response.json()["response"]
                world.append_chat_message(obj_id, "assistant", reply)
                              
            

    except Exception as e:
        print(f"error in websocket: {e}")
        clients.remove(websocket)

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