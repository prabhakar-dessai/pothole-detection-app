
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from main import *

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'png', 'jpg'}
app.config['UPLOAD_FOLDER'] = 'static/videos/input'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_view', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            processed_filename = process_file(filename)  # code in main.py
            print(f"Processed video file path: {processed_filename}")  # Debugging step
            return render_template('index.html', input_filename=filename, processed_filename=processed_filename, input_type='video', processed_type='video')
    return render_template('index.html')

@app.route('/image_view', methods=['GET', 'POST'])
def upload_file_img():
    if request.method == 'POST':
        file = request.files['img']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            processed_filename = process_file_img(filename)  # code in main.py
            print(f"Processed image file path: {processed_filename}")  # Debugging step
            return render_template('index.html', input_filename=filename, processed_filename=processed_filename, input_type='image', processed_type='image')
    return render_template('index.html')

if __name__ == "_main_":  # Corrected the condition for running the app
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    app.run()