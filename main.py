import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

reviews_datasets = pd.read_csv('Reviews.csv')
reviews_datasets = reviews_datasets.head(20000)
count_vect = CountVectorizer(max_df=0.8 , min_df=2 , stop_words='english')
doc_term_matrix = count_vect.fit_transform(reviews_datasets['Text'].values.astype('U'))

LDA = LatentDirichletAllocation(n_components=5, random_state=42)
LDA.fit(doc_term_matrix)

#The following script randomly fetches 10 words from our vocabulary:
for i in range(10):
    random_id = random.randint(0,len(count_vect.get_feature_names()))
    print(count_vect.get_feature_names()[random_id])

first_topic = LDA.components_[0]

top_topic_words = first_topic.argsort()[-10:]
print(top_topic_words)

#These indexes can then be used to retrieve the value of the words from the count_vect object,
for i in top_topic_words:
    print(count_vect.get_feature_names()[i])

#Let's print the 10 words with highest probabilities for all the five topics:
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')