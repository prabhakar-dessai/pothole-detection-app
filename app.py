from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
import os
from main import process_file

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['UPLOAD_FOLDER'] = 'static/videos/input'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/',methods=['GET','POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['video']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
            processed_filename = process_file(filename) # code in main.py
            return render_template('success.html', filename=processed_filename) # change after including processing step
    return render_template('index.html')


if __name__ == "__main__":
    app.run()