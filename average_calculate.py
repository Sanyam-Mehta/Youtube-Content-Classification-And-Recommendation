import pandas as pd
from bs4 import BeautifulSoup


csv='scored_comments.csv'
df = pd.read_csv(csv,encoding='utf-8')


cols = ['video_id','title','channel_title','category_id','tags','views','likes','dislikes','comment_total','thumbnail','date']
tf = pd.read_csv("GBvideos.csv",header=None,names=cols, low_memory=False)

df = df.groupby('video_id', as_index=False, sort=False)['polarity'].mean()

print(list(tf))

print(df)




#b = pd.read_csv("entity_ids.csv")
del tf['views']
del tf['likes']
del tf['dislikes']
del tf['comment_total']
del tf['thumbnail']
del tf['date']
merge = df.merge(tf,how='inner', on='video_id')

#merge = tf.join(df, on='video_id')

merge.drop_duplicates(subset ="video_id", 
                    keep = 'first', inplace = True) 

merge.to_csv("FINAL.csv", index = False)


