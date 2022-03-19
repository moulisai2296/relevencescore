from flask import *
from predict import *
from search_engine import *

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
@app.route("/", methods=['GET','POST'])
def initial1():
    return render_template('initial.html')

@app.route('/submit', methods =["GET", "POST"])
def submit():
    if request.method == "POST":
       search_term = request.form.get("search_term")
       product_title = request.form.get("product_title")
       example = [[product_title, search_term]]
       score = predict_test(example)[0]
       print(score)
       if type(score[0]) != 'str':
           score = round(score[0], 2)
           return "Relevance Score: " + str(score)
       else: 
           return score
    return render_template("result.html")

@app.route('/search', methods=['GET', 'POST'])
def search():
    if request.method == "POST":
        search_term = request.form.get("search_term")
        example = [search_term]
        result = search_sim(example)
        response = jsonify(result)
        return response
    return render_template("search.html")

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000, debug=True)
