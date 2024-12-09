from flask import Flask, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from src.predict import predict_image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make prediction
            try:
                class_name, confidence, probabilities = predict_image(filepath)
                
                # Format probabilities for display
                prob_list = [{"class": name, "probability": f"{prob:.2%}"} 
                           for name, prob in probabilities]
                
                return render_template('result.html',
                                     filename=filename,
                                     class_name=class_name,
                                     confidence=f"{confidence:.2%}",
                                     probabilities=prob_list)
            except Exception as e:
                return jsonify({'error': str(e)})
    
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True) 