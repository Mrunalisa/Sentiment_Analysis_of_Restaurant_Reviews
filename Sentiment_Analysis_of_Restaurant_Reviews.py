import numpy as np
import pandas as pd

#load data(file)/read file 
data=pd.read_csv('Restaurant_Reviews.tsv',delimiter="\t",quoting=3)
# print("shape of file data :",data.shape)
# print("file columns :",data.columns)
# print("file data head: ",data.head())
# print("data info : ",data.info)

#importing lab for performing nlp on resta review file dataset
#data processing
import nltk #nlp
import re #regular expression
print(nltk.download('stopwords'))
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#cleaning the reviews
corpus=[]
for i in range(0,1000):
    review = re.sub("[^a-zA-Z]"," ",data['Review'][i])
    review=review.lower()
    review_words=review.split()
    review_words=[word for word in review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review_words]
    review=' '.join(review)
    # print(corpus.append(review))
    # print(corpus[:1500])
    corpus.append(review)
    corpus[:1500]

    #creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer #countvectorizer is method to convert txt to numerical data
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,1].values

# split data in two parts(traning and testing)
from sklearn.model_selection import train_test_split #use split the original data into traning data/test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
# print(x_train.shape, x_test.shape,y_train.shape,y_test.shape)

# fitting navie bayes to traning set
from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(x_train,y_train)
ypred=classifier.predict(x_test)
# print(ypred)
print("\n\n")
#accuracy , precision and recall
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sc1=accuracy_score(y_test,ypred)
sc2=precision_score(y_test,ypred)
sc3=recall_score(y_test,ypred)

white_space = "   "

print(f'''                                        -------------------
                                    {white_space} |      SCORES     |
                                        -------------------''')
print()
print("                                     Accuracy Score : {}%".format(round(sc1*100,2)))
print("                                     Precision Score : {}%".format(round(sc2*100,2)))
print("                                     Recall Score : {}%".format(round(sc3*100,2)))
print("\n")
print("---------------------------------------------------------------------------")
print("\n")
best_accuracy=0.0
alpha_val=0.0
for i in np.arange(0.1,1.1,0.1):
    temp_classifier=MultinomialNB(alpha=i)
    temp_classifier.fit(x_train,y_train)
    temp_y_pred=temp_classifier.predict(x_test)
    score=accuracy_score(y_test,temp_y_pred)
    print("                          Accuracy Score for alpha={} is : {}%".format(round(i,1),round(score*100,2)))
    if score>best_accuracy:
        best_accuracy=score
        alpha_val=i
print("                        ----------------------------------------------")
print("                              The Best Accuracy is {}%".format(round(best_accuracy*100,2),round(alpha_val,1)))
print("                        ----------------------------------------------\n")

classifier=MultinomialNB(alpha=0.2)
classifier.fit(x_train,y_train)

def perdict_sentiment(user_review):
    user_review=re.sub("[^a-zA-Z]"," ",user_review)
    user_review=user_review.lower()
    user_review_words=user_review.split()
    user_review_words=[word for word in user_review_words if not word in set(stopwords.words('english'))]
    ps=PorterStemmer()
    final_review=[ps.stem(word) for word in user_review_words]
    final_review=' '.join(final_review)
    temp=cv.transform([final_review]).toarray()
    return classifier.predict(temp)
print(f"""{white_space}{white_space}----------------------------------------------------------------------------
      |{white_space}This Model Predicts Whether Restaurant Review is Postive or Negative   |
      ----------------------------------------------------------------------------""")
print()
while True:
    user_review=input("               Enter Your Review About Restaurant : ")
    if perdict_sentiment(user_review):
        print('                    =>This is Positive Review.')
        print("-----------------------------------------------------------------------------------------------------")
    else:
        print('                     =>This is Negative Review.')

        print("----------------------------------------------------------------------------------------------------")
    print()
    str=input("If you want to exit or quit, type 'q' or 'e'. If you don't want to stop, just press the 'enter' key : ")
    if (str == "q" or str == "e"):
        print()
        print("---------------------------------------------Exit---------------------------------------")
        print()
        break
