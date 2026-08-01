from flask import Flask, request, render_template, jsonify
import requests
from datetime import datetime, timedelta
from io import BytesIO
import base64
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from skimage import morphology 
from flask_cors import CORS
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import UNet

app = Flask(__name__)
CORS(app)

CLIENT_ID     = os.environ["SENTINEL_CLIENT_ID"]
CLIENT_SECRET = os.environ["SENTINEL_CLIENT_SECRET"]

FALSE_COLOR_SCRIPT = """
//VERSION=3
function setup() {
  return { input:["B12","B11","B04"], output:{bands:3} };
}
function evaluatePixel(sample) {
  return [sample.B12, sample.B11, sample.B04];
}
"""

def get_access_token():
    resp = requests.post(
        "https://services.sentinel-hub.com/oauth/token",
        data={
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
    )
    resp.raise_for_status()
    return resp.json()['access_token']

def build_payload(bbox, date_obj):
    start = (date_obj - timedelta(days=3)).strftime('%Y-%m-%d')
    end   = (date_obj + timedelta(days=3)).strftime('%Y-%m-%d')
    geom = {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
        ]]
    }
    return {
        "input": {
            "bounds": { "geometry": geom },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{start}T00:00:00Z",
                        "to":   f"{end}T23:59:59Z"
                    },
                    "maxCloudCoverage": 40
                }
            }]
        },
        "output": {
            "width": 512,
            "height": 512,
            "responses": [{"identifier": "default", "format": {"type": "image/jpeg"}}]
        },
        "evalscript": FALSE_COLOR_SCRIPT
    }

# Load U-Net model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
model.to(device).eval()

preprocess = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])

@app.route('/')
def index():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data     = request.get_json()
        date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
        bbox     = list(map(float, data['bbox']))

        # 1) Pull raw Sentinel-2 JPEG
        token   = get_access_token()
        headers = {'Authorization': f"Bearer {token}", 'Content-Type': 'application/json'}
        payload = build_payload(bbox, date_obj)
        resp    = requests.post(
            'https://services.sentinel-hub.com/api/v1/process',
            headers=headers, json=payload
        )
        resp.raise_for_status()

        # 2) Open it and run your U-Net
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        x   = preprocess(img).unsqueeze(0).to(device)       # [1,3,512,512]
        with torch.no_grad():
            out = model(x)                                 # [1,1,512,512]
            mask = torch.sigmoid(out).squeeze().cpu().numpy()

        # --- Apply Morphological Operations for Mask Refinement ---
        
        mask_bin = (mask > 0.5).astype(np.uint8)

        # 1. Closing: Dilation followed by Erosion. 
        selem_closing = morphology.disk(3)
        mask_bin = morphology.binary_closing(mask_bin, selem_closing).astype(np.uint8)

        # 2. Opening: Erosion followed by Dilation

        selem_opening = morphology.disk(2)
        mask_bin = morphology.binary_opening(mask_bin, selem_opening).astype(np.uint8)
        # --- End Morphological Operations ---

        
        # 3) Create a colored overlay (red, 50% alpha)
        mask_img = Image.fromarray(mask_bin * 255).resize(img.size, resample=Image.NEAREST)
        overlay  = Image.new('RGBA', img.size, (255,0,0,0))
        mpixels  = mask_img.load()
        opixels  = overlay.load()
        for y in range(img.size[1]):
            for x in range(img.size[0]):
                if mpixels[x,y] == 255: 
                    opixels[x,y] = (255,0,0,100) 
        combined = Image.alpha_composite(img.convert('RGBA'), overlay)

        # 4) Return both original and overlay as Base64 URIs
        def to_data_uri(pil_img, fmt='PNG'):
            buf = BytesIO()
            pil_img.save(buf, format=fmt)
            uri = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/{fmt.lower()};base64,{uri}"

        return jsonify({
            'sat_image': to_data_uri(img,      'JPEG'),
            'overlay'  : to_data_uri(combined,'PNG'),
        })

    except Exception as e:
        return jsonify({ 'error': str(e) }), 500


if __name__=='__main__':
    app.run(host='0.0.0.0', port=8070, debug=True)