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

food_model = YOLO('Sudanese-food-detection.pt')
cloth_model = YOLO('best.pt')

UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULTS_FILE = 'detection_results.json'
USERS_FILE = 'users.json'

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------- USER DATABASE HELPERS ----------

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def find_user(username):
    users = load_users()
    for u in users:
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


# ---------- PAGES ----------

LOGIN_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Login</title>
<style>
  body { font-family: Arial; background: #121212; color: #eee; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .box { background: #1e1e1e; padding: 30px; border-radius: 12px; width: 280px; }
  h2 { text-align: center; margin-top: 0; }
  input { width: 100%; padding: 10px; margin: 8px 0; border-radius: 8px; border: 1px solid #444; background: #2a2a2a; color: #eee; box-sizing: border-box; }
  button { width: 100%; padding: 10px; margin-top: 10px; background: #4ade80; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
  .error { color: #f87171; text-align: center; font-size: 13px; }
  .link { text-align: center; margin-top: 14px; font-size: 13px; }
  .link a { color: #4ade80; text-decoration: none; }
</style>
</head>
<body>
<div class="box">
  <h2>Login</h2>
  <form method="post" action="/login">
    <input type="text" name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Login</button>
  </form>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  <p class="link">Don't have an account? <a href="/signup">Create one</a></p>
</div>
</body>
</html>
"""

SIGNUP_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Sign up</title>
<style>
  body { font-family: Arial; background: #121212; color: #eee; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .box { background: #1e1e1e; padding: 30px; border-radius: 12px; width: 280px; }
  h2 { text-align: center; margin-top: 0; }
  input { width: 100%; padding: 10px; margin: 8px 0; border-radius: 8px; border: 1px solid #444; background: #2a2a2a; color: #eee; box-sizing: border-box; }
  button { width: 100%; padding: 10px; margin-top: 10px; background: #4ade80; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
  .error { color: #f87171; text-align: center; font-size: 13px; }
  .link { text-align: center; margin-top: 14px; font-size: 13px; }
  .link a { color: #4ade80; text-decoration: none; }
</style>
</head>
<body>
<div class="box">
  <h2>Create account</h2>
  <form method="post" action="/signup">
    <input type="text" name="username" placeholder="Username" required>
    <input type="password" name="password" placeholder="Password" required>
    <button type="submit">Sign up</button>
  </form>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  <p class="link">Already have an account? <a href="/">Login</a></p>
</div>
</body>
</html>
"""

MENU_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Choose Detector</title>
<style>
  body { font-family: Arial; background: #121212; color: #eee; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
  .box { text-align: center; }
  h2 { margin-bottom: 24px; }
  a.card { display: inline-block; background: #1e1e1e; border: 1px solid #333; border-radius: 12px; padding: 30px 40px; margin: 10px; color: #eee; text-decoration: none; width: 160px; }
  a.card:hover { border-color: #4ade80; }
</style>
</head>
<body>
<div class="box">
  <h2>Choose a detector</h2>
  <a class="card" href="/food">Food Detector</a>
  <a class="card" href="/cloth">Cloth Detector</a>
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
  body { font-family: Arial, sans-serif; background: #121212; color: #eee; margin: 0; padding: 40px 20px; }
  .container { max-width: 520px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #999; font-size: 14px; margin-bottom: 24px; }
  .back { display: block; text-align: center; color: #4ade80; margin-bottom: 20px; text-decoration: none; }
  .upload-box { background: #1e1e1e; border: 1.5px dashed #555; border-radius: 12px; padding: 30px 20px; text-align: center; }
  .btn { width: 100%; margin-top: 14px; background: #4ade80; color: #0a0a0a; border: none; border-radius: 8px; height: 44px; font-size: 15px; font-weight: 600; cursor: pointer; }
  .result-label { font-size: 13px; color: #999; margin: 24px 0 8px; }
  .result-img { width: 100%; border-radius: 12px; }
  .detections { margin-top: 16px; background: #1a1a1a; border: 0.5px solid #333; border-radius: 12px; padding: 12px 16px; }
  .det-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 0.5px solid #2a2a2a; font-size: 14px; }
  .det-coords { font-size: 11px; color: #777; margin-top: 2px; }
  .badge { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 999px; }
  .high-conf { color: #4ade80; } .mid-conf { color: #facc15; } .low-conf { color: #f87171; }
</style>
</head>
<body>
<div class="container">
  <a class="back" href="/menu">&larr; Back to menu</a>
  <h1>{{ title }}</h1>
  <p class="subtitle">{{ subtitle }}</p>

  <form method="POST" enctype="multipart/form-data">
    <div class="upload-box"><input type="file" name="image" accept="image/*" required></div>
    <button type="submit" class="btn">Detect</button>
  </form>

  {% if result_image %}
    <p class="result-label">Result</p>
    <img class="result-img" src="data:image/jpeg;base64,{{ result_image }}">
    <div class="detections">
      <p class="result-label" style="margin:0 0 10px;">Detections</p>
      {% if detections %}
        {% for d in detections %}
          <div class="det-row">
            <div>
              <div>{{ d.name }}</div>
              <div class="det-coords">x1:{{ d.box.x1 }} y1:{{ d.box.y1 }} x2:{{ d.box.x2 }} y2:{{ d.box.y2 }}</div>
            </div>
            <span class="badge {{ d.conf_class }}">{{ d.confidence }}%</span>
          </div>
        {% endfor %}
      {% else %}
        <div class="det-row"><span>No items detected</span></div>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""


# ---------- ROUTES: AUTH ----------

@app.get("/", response_class=HTMLResponse)
def login_page():
    return Template(LOGIN_PAGE).render(error=None)


@app.post("/login", response_class=HTMLResponse)
def login(username: str = Form(...), password: str = Form(...)):
    user = find_user(username)
    if user and pwd_context.verify(password, user['password_hash']):
        return RedirectResponse(url="/menu", status_code=303)
    return Template(LOGIN_PAGE).render(error="Invalid username or password")


@app.get("/signup", response_class=HTMLResponse)
def signup_page():
    return Template(SIGNUP_PAGE).render(error=None)


@app.post("/signup", response_class=HTMLResponse)
def signup(username: str = Form(...), password: str = Form(...)):
    users = load_users()

    if find_user(username):
        return Template(SIGNUP_PAGE).render(error="Username already taken")

    if len(password) < 4:
        return Template(SIGNUP_PAGE).render(error="Password must be at least 4 characters")

    hashed_password = pwd_context.hash(password)
    users.append({'username': username, 'password_hash': hashed_password})
    save_users(users)

    return RedirectResponse(url="/", status_code=303)


# ---------- ROUTES: MENU ----------

@app.get("/menu", response_class=HTMLResponse)
def menu_page():
    return MENU_PAGE


# ---------- ROUTES: FOOD DETECTOR ----------

@app.get("/food", response_class=HTMLResponse)
def food_page():
    return Template(DETECTOR_PAGE).render(
        title="Sudanese Food Detector",
        subtitle="Upload a photo to detect zalabia, cay, or mol5iya",
        result_image=None, detections=None
    )


@app.post("/food", response_class=HTMLResponse)
async def food_detect(image: UploadFile = File(...)):
    contents = await image.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'{timestamp}_{image.filename}'

    save_path = os.path.join(UPLOAD_FOLDER, image_name)
    cv2.imwrite(save_path, img)

    annotated_img, detections = run_detection(food_model, img)

    annotated_name = f'{timestamp}_detected_{image.filename}'
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)
    cv2.imwrite(annotated_path, annotated_img)

    save_result_to_json(image_name, 'food', detections)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return Template(DETECTOR_PAGE).render(
        title="Sudanese Food Detector",
        subtitle="Upload a photo to detect zalabia, cay, or mol5iya",
        result_image=result_image, detections=detections
    )


# ---------- ROUTES: CLOTH DETECTOR ----------

@app.get("/cloth", response_class=HTMLResponse)
def cloth_page():
    return Template(DETECTOR_PAGE).render(
        title="Sudanese Cloth Detector",
        subtitle="Upload a photo to detect clothing items",
        result_image=None, detections=None
    )


@app.post("/cloth", response_class=HTMLResponse)
async def cloth_detect(image: UploadFile = File(...)):
    contents = await image.read()
    file_bytes = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    image_name = f'{timestamp}_{image.filename}'

    save_path = os.path.join(UPLOAD_FOLDER, image_name)
    cv2.imwrite(save_path, img)

    annotated_img, detections = run_detection(cloth_model, img)

    annotated_name = f'{timestamp}_detected_{image.filename}'
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_name)
    cv2.imwrite(annotated_path, annotated_img)

    save_result_to_json(image_name, 'cloth', detections)

    _, buffer = cv2.imencode('.jpg', annotated_img)
    result_image = base64.b64encode(buffer).decode('utf-8')

    return Template(DETECTOR_PAGE).render(
        title="Sudanese Cloth Detector",
        subtitle="Upload a photo to detect clothing items",
        result_image=result_image, detections=detections
    )