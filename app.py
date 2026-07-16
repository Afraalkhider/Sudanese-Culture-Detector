from flask import Flask, request, render_template_string
from ultralytics import YOLO
import cv2
import base64
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
model = YOLO('Sudanese-food-detection.pt')

UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<title>Sudanese Food Detector</title>
<style>
  body { font-family: Arial, sans-serif; background: #121212; color: #eee; margin: 0; padding: 40px 20px; }
  .container { max-width: 520px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 4px; }
  .subtitle { text-align: center; color: #999; font-size: 14px; margin-bottom: 24px; }
  .upload-box { background: #1e1e1e; border: 1.5px dashed #555; border-radius: 12px; padding: 30px 20px; text-align: center; }
  .upload-box input[type=file] { margin: 10px 0; color: #ccc; }
  .btn { width: 100%; margin-top: 14px; background: #4ade80; color: #0a0a0a; border: none; border-radius: 8px; height: 44px; font-size: 15px; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #3fce70; }
  .result-label { font-size: 13px; color: #999; margin: 24px 0 8px; }
  .result-img { width: 100%; border-radius: 12px; }
  .detections { margin-top: 16px; background: #1a1a1a; border: 0.5px solid #333; border-radius: 12px; padding: 12px 16px; }
  .det-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 0.5px solid #2a2a2a; font-size: 14px; }
  .det-row:last-child { border-bottom: none; }
  .badge { font-size: 12px; font-weight: 600; padding: 2px 10px; border-radius: 999px; }
  .high-conf { color: #4ade80; }
  .mid-conf { color: #facc15; }
  .low-conf { color: #f87171; }
</style>
</head>
<body>
<div class="container">
  <h1>Sudanese Food Detector</h1>
  <p class="subtitle">Upload a photo to detect zalabia, cay, or mol5iya</p>

  <form method="POST" enctype="multipart/form-data">
    <div class="upload-box">
      <input type="file" name="image" accept="image/*" required>
    </div>
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
            <span>{{ d.name }}</span>
            <span class="badge {{ d.conf_class }}">{{ d.confidence }}%</span>
          </div>
        {% endfor %}
      {% else %}
        <div class="det-row"><span>No food detected</span></div>
      {% endif %}
    </div>
  {% endif %}
</div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result_image = None
    detections = []

    if request.method == 'POST':
        file = request.files['image']
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = os.path.join(UPLOAD_FOLDER, f'{timestamp}_{file.filename}')
        cv2.imwrite(save_path, img)

        results = model.predict(img, conf=0.5, verbose=False)
        annotated_img = results[0].plot()

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0]) * 100

            if confidence >= 80:
                conf_class = 'high-conf'
            elif confidence >= 50:
                conf_class = 'mid-conf'
            else:
                conf_class = 'low-conf'

            detections.append({
                'name': class_name,
                'confidence': round(confidence, 1),
                'conf_class': conf_class
            })

        _, buffer = cv2.imencode('.jpg', annotated_img)
        result_image = base64.b64encode(buffer).decode('utf-8')

    return render_template_string(HTML_PAGE, result_image=result_image, detections=detections)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)