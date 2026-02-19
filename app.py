import os
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms
from PIL import Image
from timm import create_model
import random

# Initialize Flask app
app = Flask(__name__)

# CORS: allow React dev server (e.g. port 8080) to call this API
@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

# Paths and Parameters
MODEL_PATH = "efficientnet_plantdoc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Minimum confidence from disease model to treat as a valid plant prediction (reject unclear / non-plant)
MIN_PLANT_CONFIDENCE = 0.50

# WHITELIST: ImageNet top‑5 must contain at least one of these to be treated as plant/leaf (otherwise we reject)
# Using top‑5 so real leaves that get top‑1 as "leaf beetle" or "cabbage butterfly" still pass
PLANT_ACCEPT_KEYWORDS = [
    "cabbage", "broccoli", "cauliflower", "zucchini", "squash", "cucumber", "artichoke",
    "pepper", "cardoon", "mushroom", "strawberry", "orange", "lemon", "fig", "pineapple",
    "banana", "jackfruit", "custard apple", "pomegranate", "hay", "daisy", "corn",
    "acorn", "hip", "buckeye", "fungus", "agaric", "gyromitra", "stinkhorn", "earthstar",
    "hen-of-the-woods", "bolete", "ear", "rapeseed", "lady's slipper", "granny",
    "greenhouse", "leaf", "vegetable", "fruit", "flower", "plant", "potato", "tomato",
]
# Number of ImageNet top predictions to check for plant keywords (any match = allow)
PLANT_CHECK_TOP_K = 5
# Fallback blacklist (used only if whitelist check is skipped): reject these
NON_PLANT_KEYWORDS = [
    "bench", "desk", "table", "chair", "couch", "bed", "sofa", "furniture",
    "computer", "phone", "keyboard", "monitor", "laptop", "television", "printer",
    "car", "vehicle", "boat", "airplane", "bicycle", "train", "ship", "bus", "minivan",
    "church", "castle", "palace", "building", "house", "home", "barn", "monastery",
    "groom", "diver", "ballplayer", "person", "dog", "cat", "bird", "fish", "reptile",
    "camera", "bookcase", "book", "telephone", "wardrobe", "refrigerator", "cabinet",
    "shelf", "mirror", "pillow", "towel", "barber", "rocking", "folding", "ashcan",
    "tower", "bridge", "window", "door", "envelope", "menu", "sign", "notebook",
    "wallet", "purse", "card", "document", "license", "certificate", "mask", "binder",
    "photo", "portrait", "credit", "id ", "digilocker", "academic", "paper", "letter",
]

# Define the disease model
NUM_CLASSES = 2  # Adjust based on your dataset (e.g., 2 for Diseased and Healthy)
model = create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# Load the trained model
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_PATH}' not found.")
    exit(1)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit(1)

# Plant-vs-non-plant checker: pretrained ImageNet model + reject list
plant_checker_model = None
imagenet_class_names = []

def _load_imagenet_classes():
    """Load ImageNet class names (1000 classes). Tries local file first, then URL."""
    # 1) Local file next to app.py (optional)
    base = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(base, "imagenet_classes.txt")
    if os.path.isfile(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
        except Exception:
            pass
    # 2) Fetch from PyTorch hub
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return [line.decode("utf-8").strip() for line in resp.readlines()]
    except Exception:
        pass
    return []

def _init_plant_checker():
    global plant_checker_model, imagenet_class_names
    try:
        plant_checker_model = create_model("efficientnet_b0", pretrained=True, num_classes=1000)
        plant_checker_model = plant_checker_model.to(DEVICE)
        plant_checker_model.eval()
        imagenet_class_names = _load_imagenet_classes()
        if not imagenet_class_names:
            print("Plant checker: ImageNet class names not loaded; using confidence-only validation.")
        else:
            print("Plant checker loaded (ImageNet). Non-plant images will be rejected.")
    except Exception as e:
        print(f"Plant checker not loaded ({e}). Using confidence-only validation.")

_init_plant_checker()

def _is_likely_non_plant(image_tensor):
    """Return True if the image should be REJECTED (not a plant/leaf). Uses whitelist on top‑K: allow if any of top‑K looks like plant."""
    global plant_checker_model, imagenet_class_names
    if plant_checker_model is None:
        return False  # skip check
    with torch.no_grad():
        logits = plant_checker_model(image_tensor)
        # Get top‑K predicted class indices (K = PLANT_CHECK_TOP_K)
        k = min(PLANT_CHECK_TOP_K, logits.shape[1])
        _, top_k = torch.topk(logits, k, dim=1)
        top_indices = top_k[0].tolist()
    if not imagenet_class_names:
        return False
    for idx in top_indices:
        if idx >= len(imagenet_class_names):
            continue
        label = imagenet_class_names[idx].lower()
        for kw in PLANT_ACCEPT_KEYWORDS:
            if kw in label:
                return False  # at least one of top‑K looks like plant, allow
    # None of top‑K is in plant whitelist -> reject (document, face, bench, etc.)
    return True

# Define transformations for the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Nutrient scoring function (mock implementation)
def get_nutrient_score(image_tensor):
    """
    Mock implementation to generate a nutrient score.
    Replace with your logic for nutrient deficiency prediction.
    """
    return torch.randint(70, 100, (1,)).item()

# Confidence score function (random between 1 to 10)
def get_confidence_score():
    return round(random.uniform(1, 10), 2)

# User-facing message when image is not a plant/leaf
NOT_PLANT_ERROR = (
    "This doesn't look like a plant or leaf image. "
    "Please upload a clear photo of a plant leaf for disease detection."
)
LOW_CONFIDENCE_ERROR = (
    "Unable to recognize a plant leaf in this image. "
    "Please upload a clear, close-up photo of a plant leaf."
)

# Prediction function
def predict(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # Reject obvious non-plant images (bench, house, person, etc.) using ImageNet
        if _is_likely_non_plant(image_tensor):
            return {"error": NOT_PLANT_ERROR}

        # Test-time augmentation (TTA): original + horizontal flip, then average softmax
        # Improves accuracy and confidence, especially for diseased leaves
        with torch.no_grad():
            logits_1 = model(image_tensor)
            image_flipped = torch.flip(image_tensor, dims=[-1])  # horizontal flip
            logits_2 = model(image_flipped)
            # Average logits then softmax (more stable than averaging softmax)
            avg_logits = (logits_1 + logits_2) / 2.0
            confidence_scores = torch.softmax(avg_logits, dim=1)
            _, predicted = torch.max(confidence_scores, 1)
        
        # Decode the results (order must match how the model was trained)
        # Many PlantDoc/training scripts use 0=Diseased, 1=Healthy (e.g. folder order)
        CLASS_NAMES = ['Diseased', 'Healthy']  # index 0 = Diseased, index 1 = Healthy
        pred_idx = predicted.item()
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = confidence_scores[0, pred_idx].item()

        # Reject low-confidence predictions (unclear or non-plant images)
        if confidence < MIN_PLANT_CONFIDENCE:
            return {"error": LOW_CONFIDENCE_ERROR}
        
        # Get nutrient score
        nutrient_score = get_nutrient_score(image_tensor)
        conf_pct = round(confidence * 100, 2)

        # Human-friendly message and confidence tier for the UI
        if predicted_class == "Healthy":
            message = "No disease detected. Your leaf appears healthy."
            recommendation = "Keep monitoring; ensure good light and water."
        else:
            message = "Disease indicators detected on the leaf."
            recommendation = "For a specific diagnosis and treatment, consult a plant expert or use a clearer, close-up photo of the affected area."

        if conf_pct >= 85:
            confidence_tier = "high"
        elif conf_pct >= 60:
            confidence_tier = "moderate"
        else:
            confidence_tier = "low"

        return {
            "class": predicted_class,
            "confidence": conf_pct,
            "message": message,
            "recommendation": recommendation,
            "confidence_tier": confidence_tier,
            "nutrient_score": nutrient_score,
            "random_confidence_score": get_confidence_score()
        }
    except Exception as e:
        return {"error": str(e)}

# Front end: React app (leaf-doctor-frontend-main). Build with: cd leaf-doctor-frontend-main && npm run build
# Then this server serves the built UI from dist/ and the API from /predict.
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "leaf-doctor-frontend-main", "dist")

def _serve_frontend(path=""):
    if path and os.path.exists(os.path.join(FRONTEND_DIST, path)):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")

@app.route('/')
def index():
    if os.path.isdir(FRONTEND_DIST):
        return _serve_frontend("index.html")
    # Fallback: old template if React app not built yet
    from flask import render_template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    # Front end sends the file with key "image" (see leaf-doctor-frontend-main/src/lib/api.ts)
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    if file.filename == '' or not file.filename:
        return jsonify({"error": "No image selected"}), 400
    try:
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)
        # Save with a unique name to avoid collisions
        import tempfile
        _, ext = os.path.splitext(file.filename)
        fd, image_path = tempfile.mkstemp(suffix=ext or ".jpg", dir=uploads_dir)
        try:
            file.save(image_path)
            result = predict(image_path)
            if "error" in result:
                return jsonify(result), 400
            return jsonify(result)
        finally:
            try:
                os.close(fd)
                if os.path.exists(image_path):
                    os.remove(image_path)
            except OSError:
                pass
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# SPA: serve React app for any other path (e.g. /assets/..., or client-side routes)
@app.route('/<path:path>')
def serve_spa(path):
    if not os.path.isdir(FRONTEND_DIST):
        return {"error": "Front end not built. Run: cd leaf-doctor-frontend-main && npm run build"}, 404
    return _serve_frontend(path)

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred while starting the server: {e}")
