from flask import Flask, render_template, request
from recommender_nmf import recommend_nmf


app = Flask(__name__)

@app.route('/')
def hello_fun():
    return render_template("main.html", title1="Cascabel")

@app.route('/recommendations')
def recommender():
    user_input_from_app = request.args
    # Using nmf_mode together with user_input_dict to give recommendations to user.
    top10_films = recommend_nmf(user_input_from_app)
    return render_template("recommender.html", films_var = top10_films)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
