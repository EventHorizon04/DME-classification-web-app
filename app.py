from flask import Flask, render_template, request, jsonify
import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import pickle
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff'}

# Init web
app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'test_web_app'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32MB

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'oct'), exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'map'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
# Load trained model
model_path = 'vgg-xgb.pkl'
model = pickle.load(open(model_path, "rb"))

# Preprocessing function to prepare the image for VGG-19
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_feature(image):
   
    # Load the VGG-19 model pre-trained on ImageNet, exclude the fully connected layers
    vgg19 = models.vgg19(weights='IMAGENET1K_V1').features.to(device)
    vgg19.eval()  # Set the model to evaluation mode

    # Load the image and apply preprocessing
    image = image.convert('RGB')
    image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass to extract features
    with torch.no_grad():
        features = vgg19(image)
    return features.flatten().cpu().numpy()  # Convert to numpy array and flatten


def make_prediction(oct_img, map_img):
    """Make prediction using both images"""
    try:
        # Extract features
        oct_features = extract_feature(oct_img)
        map_features = extract_feature(map_img)

        # Combine features
        combined_features = np.hstack((map_features, oct_features))
        
        # Make prediction
        prediction = model.predict_proba(combined_features.reshape(1, -1))[0]
        return prediction
    except Exception as e:
        return None, f"Prediction error: {str(e)}"


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/uploads", methods=['GET','POST'])
def upload_file():
    try:
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'success': False, 'error':'Please upload both images.'}), 400
        
        oct_file = request.files['image1']
        map_file = request.files['image2'] 
        
        # Check if files are selected
        if oct_file.filename == '' or map_file.filename == '':
            return jsonify({'success': False, 'error':'No file selected.'}), 400
        # Validate file types
        if not (allowed_file(oct_file.filename) and allowed_file(map_file.filename)):
            return jsonify({'success': False, 'error':'Invalid file type. Please upload JPG or PNG files.'}), 400
        
        oct_path = os.path.join(app.config['UPLOAD_FOLDER'], 'oct', oct_file.filename)
        map_path = os.path.join(app.config['UPLOAD_FOLDER'], 'map', map_file.filename)
        
        oct_file.save(oct_path)
        map_file.save(map_path)
        

        try:
            oct_image = Image.open(oct_path)
            map_image = Image.open(map_path)
            print("Images loaded successfully.")
        except Exception as e:
            print(f"Error loading images: {e}")
            return jsonify({'success': False, 'error': 'Error loading images. Please ensure they are valid image files.'}), 400
        
        prediction = make_prediction(oct_image, map_image)
        print(prediction)
        if prediction is None:
            # Cleanup before returning error
            os.remove(oct_path)
            os.remove(map_path)
            return jsonify({'success': False, 'error': 'Prediction error occurred.'}), 500
        
        response_prob, non_response_prob = prediction[0], prediction[1]
        os.remove(oct_path)
        os.remove(map_path)
        return jsonify({
            'success': True,
            'response_probability': float(response_prob),
            'non_response_probability': float(non_response_prob)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': f'An unexpected error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    # app.run(debug=True)