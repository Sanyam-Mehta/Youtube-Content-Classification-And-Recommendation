import flask_blog
import pandas as pd

def getVideos(output_cat_id) :
	df = pd.read_csv("/home/nikita/Desktop/FlaskProject/final_average_score.csv", delimiter = ',')
	df = df.loc[df['category_id'] == str(output_cat_id)]
	df = df.sort_values('polarity', ascending = False)
	final_df = pd.DataFrame()
	final_df['VIDEO TITLE'] = df['title']
	final_df['CHANNEL TITLE'] = df['channel_title']
	final_df['USER RATING'] = df['polarity']
	final_df.reset_index(drop=True, inplace=True)
	return final_df.head(5)
	
