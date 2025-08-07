from flask import Flask, render_template, request
from youtube_predictor import predict_youtube_category

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_label = None
    url = None  # initialize URL
    if request.method == 'POST':
        url = request.form.get('youtube_url')
        predicted_label = predict_youtube_category(url)
    return render_template('index.html', prediction=predicted_label, youtube_url=url)

if __name__ == '__main__':
    app.run(debug=True)
