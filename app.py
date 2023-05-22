from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config['ALLOWED_EXTENSIONS'] = {'mp4'}
app.config['UPLOAD_FOLDER'] = 'static/videos/'

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
            # filename = secure_filename(file.filename) #not working rn
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        #     processed_filename = process_file(filename)
            return render_template('success.html', filename=file.filename) # change after including processing step
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)