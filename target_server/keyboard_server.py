import redis
import numpy as np
from pynput import keyboard
from threading import Lock
import pickle
import time
vlock = Lock()
v = np.zeros((3,13), dtype=np.float32)
def on_press(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.up:
            v[:,0] = 0.02
        elif key == keyboard.Key.down:
            v[:,0] = -0.02
        elif key == keyboard.Key.left:
            v[:,1] = -0.02
        elif key == keyboard.Key.right:
            v[:,1] = 0.02
        elif key == keyboard.Key.shift:
            v[:,2] = 0.02
        elif key == keyboard.Key.ctrl:
            v[:,2] = -0.02
        elif key.char == "w":
            v[1:,0] = 0.01
        elif key.char == "s":
            v[1:,0] = -0.01
        elif key.char == "q":
            v[1,1] = -0.01
            v[2,1] = 0.01
        elif key.char == "e":
            v[1,1] = 0.01
            v[2,1] = -0.01
        elif key.char == "a":
            v[1:,2] = -0.01
        elif key.char == "d":
            v[1:,2] = 0.01
    except:
        pass
    vlock.release()

def on_release(key):
    vlock.acquire()
    try:
        if key == keyboard.Key.up:
            v[:,0] = 0.0
        elif key == keyboard.Key.down:
            v[:,0] = 0.0
        elif key == keyboard.Key.left:
            v[:,1] = 0.0
        elif key == keyboard.Key.right:
            v[:,1] = 0.0
        elif key == keyboard.Key.shift:
            v[:,2] = 0.0
        elif key == keyboard.Key.ctrl:
            v[:,2] = 0.0
        elif key.char == "w":
            v[1:,0] = 0.0
        elif key.char == "s":
            v[1:,0] = 0.0
        elif key.char == "q":
            v[1,1] = 0.0
            v[2,1] = 0.0
        elif key.char == "e":
            v[1,1] = 0.0
            v[2,1] = 0.0
        elif key.char == "a":
            v[1:,2] = 0.0
        elif key.char == "d":
            v[1:,2] = 0.0
    except:
        pass
    vlock.release()

redis_client = redis.Redis(host='localhost', port=6379, db=0)

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

target = pickle.loads(redis_client.get("current_target"))
while target is not None:
    vlock.acquire()
    target += v
    vlock.release()
    print("Target:", target)
    redis_client.set("target", pickle.dumps(target))
    redis_client.set("server_ready", "true")
    time.sleep(0.03)
