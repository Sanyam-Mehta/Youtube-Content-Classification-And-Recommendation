from flask import Flask, render_template, url_for, redirect, flash, request
from forms import RegistrationForm
import os
from sklearn.naive_bayes import MultinomialNB
import GetCategory
import getResults
app = Flask(__name__)

app.config['SECRET_KEY'] = '1234'

inputstring = ""
global outputCategory 

# @app.route("/results")
# def results():
# 	return render_template('results.html', outputCategory = outputCategory)


@app.route("/run")
def run():
	GetCategory.run()
	return "<h1>Success</h1>"

@app.route("/register", methods = ['GET', 'POST'])
def register():
	form = RegistrationForm()
	if form.validate_on_submit():
		global outputCategory
		inputstring = form.SearchQuery.data
		outputCategory = GetCategory.getCategory(inputstring)
		print("output category:", outputCategory)
		dataframe = getResults.getVideos(outputCategory)
		print(dataframe)
		outputCategoryName = GetCategory.out
		return render_template('results.html', outputCategory = outputCategory, dataframe = dataframe.to_html(), outputCategoryName = outputCategoryName)
	else :	
		return render_template('register.html', form = form) 

if __name__ == "__main__":
	app.run("localhost",debug = True)