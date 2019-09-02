#!/usr/bin/env python
# coding: utf-8

# In[188]:


from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[189]:


tokenizer = RegexpTokenizer(r'\w+')
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()


# In[190]:


def getStemmedReview(review):
    review=review.lower()
    review = review.replace("<br /><br />"," ")
    
    tokens = tokenizer.tokenize(review)
    stopwords = [token for token in tokens if token not in en_stopwords]
    new_token = [ps.stem(tokens) for tokens in stopwords]
    
    cleaned_review = " ".join(new_token)
    
    return cleaned_review


# In[191]:


sample = """I loved this movie <br /><br /> since I was 7 and I saw it on the opening day. It was so touching and beautiful. I strongly recommend seeing for all. It's a movie to watch with your family by far.<br /><br />My MPAA rating: PG-13 for thematic elements, prolonged scenes of disastor, nudity/sexuality and some language."""


# In[192]:


getStemmedReview(sample)


# In[193]:


x = ["This was an awesome movie",
     "Great good movie! I liked it a lot",
     "Happy Ending! awesome acting by the hero",
     "loved it! truly great",
     "bad not upto the mark",
     "could have better",
     "did not like the movie",
     "Surely a Disappointing movie"]
y = [1,1,1,1,0,0,0,0]


# In[194]:


cleaned_x =[]
for i in x:
    cleaned_x.append(getStemmedReview(i))
print(cleaned_x)


# In[195]:


# Vectorisation
from sklearn.feature_extraction.text import CountVectorizer
cv= CountVectorizer()
vectorized_x = (cv.fit_transform(cleaned_x)).toarray()
print(vectorized_x)
print(cv.get_feature_names())
print(vectorized_x.shape)
print(cv.vocabulary_)


# In[206]:


# Applying Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB

test_x = ["The movie I saw was good", "I was happy and I loved the acting in the movie"]

mnb= MultinomialNB()
#Training
mnb.fit(vectorized_x,y)
#prediction
test_clean = [getStemmedReview(i) for i in test_x]
xt_vec = cv.transform(test_clean).toarray()
#print(test_clean)
print (mnb.predict(xt_vec))


# In[207]:


for i in mnb.predict(xt_vec):
    if i==1:
        print("Positive Review")
    else:
        print("Negative Review")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




