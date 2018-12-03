from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import Length, DataRequired
class RegistrationForm(FlaskForm):
	SearchQuery = StringField('SearchQuery', validators = [DataRequired(), Length(min = 2, max = 20)])
	submit = SubmitField('Search_Videos')