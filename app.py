from flask import Flask, render_template, request, redirect
import Caption_It

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def home_updated():
    if request.method == 'POST':
        f = request.files['userfile']
        path = "./static/{}".format(f.filename)
        f.save(path)
        caption = Caption_It.caption_this_image(path)
        #print(caption)
        result_dict = {
            'image':path,
            'caption':caption
        }
    return render_template("index.html", your_result = result_dict)


if __name__=='__main__':
    app.run(debug=True)