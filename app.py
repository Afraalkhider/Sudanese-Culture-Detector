from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from ultralytics import YOLO
from passlib.context import CryptContext
import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
from jinja2 import Template

app = FastAPI()

# Lazy-loaded models: they start as None and only load into memory
# the first time they're actually needed, instead of at server startup.
food_model = None
cloth_model = None


def get_food_model():
    global food_model
    if food_model is None:
        food_model = YOLO('Sudanese-food-detection.pt')
    return food_model


def get_cloth_model():
    global cloth_model
    if cloth_model is None:
        cloth_model = YOLO('best.pt')
    return cloth_model


UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULTS_FILE = 'detection_results.json'
USERS_FILE = 'users.json'

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def find_user(username):
    for u in load_users():
        if u['username'] == username:
            return u
    return None


def save_result_to_json(image_name, model_type, detections):
    record = {
        'image': image_name,
        'model': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': detections
    }
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(record)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def run_detection(model, img):
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_img = results[0].plot()

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0]) * 100
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if confidence >= 80:
            conf_class = 'high-conf'
        elif confidence >= 50:
            conf_class = 'mid-conf'
        else:
            conf_class = 'low-conf'

        detections.append({
            'name': class_name,
            'confidence': round(confidence, 1),
            'conf_class': conf_class,
            'box': {
                'x1': round(x1, 1), 'y1': round(y1, 1),
                'x2': round(x2, 1), 'y2': round(y2, 1)
            }
        })

    return annotated_img, detections


# ---------- SHARED STYLE: EARTHY / SUDANESE THEME ----------

BASE_STYLE = """
:root {
  --sand: #f2e6d3;
  --sand-dark: #e6d3b3;
  --clay: #b5652d;
  --clay-dark: #8a4a1f;
  --coffee: #4a3222;
  --coffee-light: #6b4a35;
  --gold: #c9932f;
}
body {
  font-family: 'Georgia', 'Amiri', serif;
  background: var(--sand);
  color: var(--coffee);
  margin: 0;
}
"""

PATTERN_BG = """
background-image:
  radial-gradient(circle at 20% 20%, rgba(181,101,45,0.08) 0, transparent 40%),
  radial-gradient(circle at 80% 80%, rgba(201,147,47,0.10) 0, transparent 40%),
  repeating-linear-gradient(45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px),
  repeating-linear-gradient(-45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px);
"""


# ---------- LOGIN PAGE ----------

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Login</title>
<style>
""" + BASE_STYLE + """
.hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  """ + PATTERN_BG + """
}
.card {
  background: #fffaf2;
  border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay);
  border-radius: 14px;
  padding: 36px 32px;
  width: 300px;
  box-sizing: border-box;
}
.brand { text-align: center; margin-bottom: 6px; }
.brand-icon { font-size: 30px; color: var(--clay); }
h2 { text-align: center; margin: 6px 0 2px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 13px; margin: 0 0 24px; }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0;
  border-radius: 8px; border: 1px solid var(--sand-dark);
  background: #fff; color: var(--coffee); box-sizing: border-box;
  font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px;
  background: var(--clay); color: #fff; border: none;
  border-radius: 8px; font-size: 14px; cursor: pointer;
  font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <div class="brand"><i class="ti ti-viewfinder brand-icon"></i></div>
    <h2>SudanScan</h2>
    <p class="subtitle">Sudanese Food & Cloth Detection System</p>
    <form method="post" action="/login">
      <input type="text" name="username" placeholder="Username" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Login</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Don't have an account? <a href="/signup">Create one</a></p>
  </div>
</div>
</body>
</html>
"""

SIGNUP_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Sign up</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card {
  background: #fffaf2; border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay); border-radius: 14px;
  padding: 36px 32px; width: 300px; box-sizing: border-box;
}
h2 { text-align: center; margin: 0 0 20px; font-weight: normal; color: var(--coffee); }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px;
  border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee);
  box-sizing: border-box; font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px; background: var(--clay);
  color: #fff; border: none; border-radius: 8px; font-size: 14px;
  cursor: pointer; font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Create account</h2>
    <form method="post" action="/signup">
      <input type="text" name="username" placeholder="Username" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Sign up</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Already have an account? <a href="/">Login</a></p>
  </div>
</div>
</body>
</html>
"""

MENU_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Choose detector</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.wrap { text-align: center; }
h2 { font-weight: normal; margin-bottom: 6px; color: var(--coffee); }
.subtitle { color: var(--coffee-light); font-size: 13px; margin: 0 0 28px; }
.cards { display: flex; gap: 16px; justify-content: center; }
a.card {
  display: block; background: #fffaf2; border: 1px solid var(--sand-dark);
  border-radius: 14px; padding: 32px 40px; text-decoration: none;
  color: var(--coffee); width: 150px; transition: border-color 0.15s;
}
a.card:hover { border-color: var(--clay); }
a.card i { font-size: 26px; color: var(--clay); }
a.card p { margin: 12px 0 0; font-size: 14px; }
</style>
</head>
<body>
<div class="hero">
  <div class="wrap">
    <h2>Choose a detector</h2>
    <p class="subtitle">SudanScan</p>
    <div class="cards">
      <a class="card" href="/food"><i class="ti ti-soup"></i><p>Food detector</p></a>
      <a class="card" href="/cloth"><i class="ti ti-shirt"></i><p>Cloth detector</p></a>
    </div>
  </div>
</div>
</body>
</html>
"""

DETECTOR_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>{{ title }}</title>
<style>
""" + BASE_STYLE + """
.container { max-width: 520px; margin: 0 auto; padding: 40px 20px; }
.back { display: block; text-align: center; color: var(--clay); margin-bottom: 20px; text-decoration: none; font-size: 13px; }
h1 { text-align: center; margin-bottom: 4px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 14px; margin-bottom: 24px; }
.upload-box { background: #fffaf2; border: 1.5px dashed var(--sand-dark); border-radius: 12px; padding: 30px 20px; text-align: center; }
.upload-box i { font-size: 24px; color: var(--clay); }
.btn { width: 100%; margin-top: 14px; background: var(--clay); color: #fff; border: none; border-radius: 8px; height: 44px; font-size: 15px; cursor: pointer; font-family: inherit; }
.btn:hover { background: var(--clay-dark); }
.result-label { font-size: 13px; color: var(--coffee-light); margin: 24px 0 8px; }
.frame { position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--sand-dark); }
.result-img { width: 100%; display: block; }
.detections { margin-top: 16px; background: #fffaf2; border: 1px solid var(--sand-dark); border-radius: 12px; padding: 12px 16px; }
.det-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--sand); font-size: 14px; }
.det-row:last-child { border-bottom: none; }
.det-coords { font-size: 11px; color: var(--coffee-light); margin-top:from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from ultralytics import YOLO
from passlib.context import CryptContext
import cv2
import numpy as np
import base64
import os
import json
from datetime import datetime
from jinja2 import Template

app = FastAPI()

# Lazy-loaded models: they start as None and only load into memory
# the first time they're actually needed, instead of at server startup.
food_model = None
cloth_model = None


def get_food_model():
    global food_model
    if food_model is None:
        food_model = YOLO('Sudanese-food-detection.pt')
    return food_model


def get_cloth_model():
    global cloth_model
    if cloth_model is None:
        cloth_model = YOLO('best.pt')
    return cloth_model


UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULTS_FILE = 'detection_results.json'
USERS_FILE = 'users.json'

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def find_user(username):
    for u in load_users():
        if u['username'] == username:
            return u
    return None


def save_result_to_json(image_name, model_type, detections):
    record = {
        'image': image_name,
        'model': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': detections
    }
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []
    all_results.append(record)
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


def run_detection(model, img):
    results = model.predict(img, conf=0.5, verbose=False)
    annotated_img = results[0].plot()

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0]) * 100
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        if confidence >= 80:
            conf_class = 'high-conf'
        elif confidence >= 50:
            conf_class = 'mid-conf'
        else:
            conf_class = 'low-conf'

        detections.append({
            'name': class_name,
            'confidence': round(confidence, 1),
            'conf_class': conf_class,
            'box': {
                'x1': round(x1, 1), 'y1': round(y1, 1),
                'x2': round(x2, 1), 'y2': round(y2, 1)
            }
        })

    return annotated_img, detections


# ---------- SHARED STYLE: EARTHY / SUDANESE THEME ----------

BASE_STYLE = """
:root {
  --sand: #f2e6d3;
  --sand-dark: #e6d3b3;
  --clay: #b5652d;
  --clay-dark: #8a4a1f;
  --coffee: #4a3222;
  --coffee-light: #6b4a35;
  --gold: #c9932f;
}
body {
  font-family: 'Georgia', 'Amiri', serif;
  background: var(--sand);
  color: var(--coffee);
  margin: 0;
}
"""

PATTERN_BG = """
background-image:
  radial-gradient(circle at 20% 20%, rgba(181,101,45,0.08) 0, transparent 40%),
  radial-gradient(circle at 80% 80%, rgba(201,147,47,0.10) 0, transparent 40%),
  repeating-linear-gradient(45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px),
  repeating-linear-gradient(-45deg, rgba(74,50,34,0.04) 0 2px, transparent 2px 26px);
"""


# ---------- LOGIN PAGE ----------

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Login</title>
<style>
""" + BASE_STYLE + """
.hero {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  """ + PATTERN_BG + """
}
.card {
  background: #fffaf2;
  border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay);
  border-radius: 14px;
  padding: 36px 32px;
  width: 300px;
  box-sizing: border-box;
}
.brand { text-align: center; margin-bottom: 6px; }
.brand-icon { font-size: 30px; color: var(--clay); }
h2 { text-align: center; margin: 6px 0 2px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 13px; margin: 0 0 24px; }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0;
  border-radius: 8px; border: 1px solid var(--sand-dark);
  background: #fff; color: var(--coffee); box-sizing: border-box;
  font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px;
  background: var(--clay); color: #fff; border: none;
  border-radius: 8px; font-size: 14px; cursor: pointer;
  font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <div class="brand"><i class="ti ti-viewfinder brand-icon"></i></div>
    <h2>SudanScan</h2>
    <p class="subtitle">Sudanese Food & Cloth Detection System</p>
    <form method="post" action="/login">
      <input type="text" name="username" placeholder="Username" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Login</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Don't have an account? <a href="/signup">Create one</a></p>
  </div>
</div>
</body>
</html>
"""

SIGNUP_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Sign up</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; """ + PATTERN_BG + """ }
.card {
  background: #fffaf2; border: 1px solid var(--sand-dark);
  border-top: 4px solid var(--clay); border-radius: 14px;
  padding: 36px 32px; width: 300px; box-sizing: border-box;
}
h2 { text-align: center; margin: 0 0 20px; font-weight: normal; color: var(--coffee); }
input {
  width: 100%; padding: 11px 12px; margin: 8px 0; border-radius: 8px;
  border: 1px solid var(--sand-dark); background: #fff; color: var(--coffee);
  box-sizing: border-box; font-family: inherit; font-size: 14px;
}
input:focus { outline: none; border-color: var(--clay); }
button {
  width: 100%; padding: 11px; margin-top: 12px; background: var(--clay);
  color: #fff; border: none; border-radius: 8px; font-size: 14px;
  cursor: pointer; font-family: inherit;
}
button:hover { background: var(--clay-dark); }
.error { color: #a3372d; text-align: center; font-size: 13px; margin-top: 10px; }
.link { text-align: center; margin-top: 16px; font-size: 13px; color: var(--coffee-light); }
.link a { color: var(--clay); text-decoration: none; }
</style>
</head>
<body>
<div class="hero">
  <div class="card">
    <h2>Create account</h2>
    <form method="post" action="/signup">
      <input type="text" name="username" placeholder="Username" required>
      <input type="password" name="password" placeholder="Password" required>
      <button type="submit">Sign up</button>
    </form>
    {% if error %}<p class="error">{{ error }}</p>{% endif %}
    <p class="link">Already have an account? <a href="/">Login</a></p>
  </div>
</div>
</body>
</html>
"""

MENU_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>SudanScan - Choose detector</title>
<style>
""" + BASE_STYLE + """
.hero { min-height: 100vh; display: flex; align-items: center; justify-content: center; }
.wrap { text-align: center; }
h2 { font-weight: normal; margin-bottom: 6px; color: var(--coffee); }
.subtitle { color: var(--coffee-light); font-size: 13px; margin: 0 0 28px; }
.cards { display: flex; gap: 16px; justify-content: center; }
a.card {
  display: block; background: #fffaf2; border: 1px solid var(--sand-dark);
  border-radius: 14px; padding: 32px 40px; text-decoration: none;
  color: var(--coffee); width: 150px; transition: border-color 0.15s;
}
a.card:hover { border-color: var(--clay); }
a.card i { font-size: 26px; color: var(--clay); }
a.card p { margin: 12px 0 0; font-size: 14px; }
</style>
</head>
<body>
<div class="hero">
  <div class="wrap">
    <h2>Choose a detector</h2>
    <p class="subtitle">SudanScan</p>
    <div class="cards">
      <a class="card" href="/food"><i class="ti ti-soup"></i><p>Food detector</p></a>
      <a class="card" href="/cloth"><i class="ti ti-shirt"></i><p>Cloth detector</p></a>
    </div>
  </div>
</div>
</body>
</html>
"""

DETECTOR_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>{{ title }}</title>
<style>
""" + BASE_STYLE + """
.container { max-width: 520px; margin: 0 auto; padding: 40px 20px; }
.back { display: block; text-align: center; color: var(--clay); margin-bottom: 20px; text-decoration: none; font-size: 13px; }
h1 { text-align: center; margin-bottom: 4px; font-weight: normal; color: var(--coffee); }
.subtitle { text-align: center; color: var(--coffee-light); font-size: 14px; margin-bottom: 24px; }
.upload-box { background: #fffaf2; border: 1.5px dashed var(--sand-dark); border-radius: 12px; padding: 30px 20px; text-align: center; }
.upload-box i { font-size: 24px; color: var(--clay); }
.btn { width: 100%; margin-top: 14px; background: var(--clay); color: #fff; border: none; border-radius: 8px; height: 44px; font-size: 15px; cursor: pointer; font-family: inherit; }
.btn:hover { background: var(--clay-dark); }
.result-label { font-size: 13px; color: var(--coffee-light); margin: 24px 0 8px; }
.frame { position: relative; border-radius: 12px; overflow: hidden; border: 1px solid var(--sand-dark); }
.result-img { width: 100%; display: block; }
.detections { margin-top: 16px; background: #fffaf2; border: 1px solid var(--sand-dark); border-radius: 12px; padding: 12px 16px; }
.det-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--sand); font-size: 14px; }
.det-row:last-child { border-bottom: none; }
.det-coords { font-size: 11px; color: var(--coffee-light); margin-top: