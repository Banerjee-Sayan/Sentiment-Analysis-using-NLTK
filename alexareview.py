# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 00:55:55 2024

@author: KIIT
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
import io
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.metrics import accuracy_score


def set_header_as_first_row(data):

    if not data.columns.size:
        data.columns = data.iloc[0]
        data = data[1:]
    return data

#Tite and subheader
st.title('Exploratory Data Analysis')
st.subheader('Data Information')

#Upload File
upload = st.file_uploader('Choose a CSV file', type='csv')
if upload is not None:
    data = pd.read_csv(upload)
    data = set_header_as_first_row(data)

#Show Dataset
    if st.checkbox('Show Dataset'):

        #Total number of rows in the dataset is stored in n
        n = data.shape[0]
        #showing the number of rows
        st.write('Number of Rows:', n)

        #Giving users option to get required number of rows
        number = st.number_input('Number of Rows to View', 1, n)
        st.dataframe(data.head(number))

    #Show datatype of each column
    if st.checkbox('Show Datatype'):
        st.text('Data Type')
        st.write(data.dtypes)
    #Show the number of rows and columns using radio button
    if st.checkbox('Show Shape'):
        data_shape = st.radio('Shape of Dataset', ('Rows', 'Columns'))
        if data_shape == 'Rows':
            st.write('Number of Rows:', data.shape[0])
        if data_shape == 'Columns':
            st.write('Number of Columns:', data.shape[1])
    #Show NULL values
    if st.checkbox('Show NULL Values'):
        st.write(data.isnull().sum())
        test = data.isnull().values.any()
        if test:
            st.warning('There are NULL values in the dataset')
            #Plotting the NULL values
            if st.checkbox('Plot NULL values'):
                fig, ax = plt.subplots()
                sns.heatmap(data.isnull(), cbar=False, ax=ax)
                st.pyplot(fig)


            # Dropping NULL values
            if st.checkbox('Drop NULL values'):
                if st.checkbox('Drop inplace'):
                    data.dropna(inplace=True)
                    st.success('NULL values dropped inplace.')
                if st.checkbox("Create a new dataframe"):
                    new_data = data.dropna()
                    st.write('A new DataFrame without NULL values has been created.')
                    st.write(new_data)
        else:
            st.write('There are no NULL values in the dataset.')

    #Finding duplicate values 
    if st.checkbox('Show Duplicate Values'):
        test1 = data.duplicated().any()
        if test1 == True:
            st.warning('There are duplicate values in the dataset')
            #Showing the duplicate values
            st.write(data[data.duplicated()])
            dup = st.selectbox("Do you want to drop duplicate values", ('Yes', 'No'))
            if dup == 'Yes':
                data.drop_duplicates()
                st.write('Duplicate values are dropped')
            else:
                st.write('Duplicate values are not dropped')
        else:
            st.success('There are no duplicate values in the dataset') 
            
    #Show Value Counts
    if st.checkbox('Show Value Counts'):
        column = st.selectbox('Select Column', data.columns)
        st.write(data[column].value_counts())
    #Show Summary
    if st.checkbox('Show Summary'):
        st.write(data.describe())
    
    #Count number of ratings in a bar chart
    if st.checkbox('Show count of ratings'):
        st.write(data.rating.value_counts())
        st.write('Bar Chart')
        st.bar_chart(data.rating.value_counts())

    #Percentage values of ratings
    if st.checkbox('Show Percentage of Ratings'):
        st.write(data.rating.value_counts(normalize=True) * 100)

    #Analysis of 'feedback' column
    if st.checkbox('Show feedback Analysis'):
        st.write(data.groupby('feedback').describe())
        st.write(data['feedback'].value_counts())
        # The below code shows that we are analyzing a random row understand which type of feedback is labeled as 1 and which is labeled as 0.
        review0 = data[data['feedback']==0].iloc[1]['verified_reviews']
        #iloc[1] is used to get a random review from the dataset where feedback is 0
        review1 = data[data['feedback']==1].iloc[1]['verified_reviews']
        st.write('Random review to classify feedback 0 and 1')
        st.write('Feedback 0:', review0)
        st.write('Feedback 1:', review1)
        st.write('Feedback 0 indicates negative review and Feedback 1 indicates positive review')
        st.write('Visualizing positive and negative feedbacks')
        feedback_data = data['feedback'].tolist()
        positive_feedback = feedback_data.count(1)
        negative_feedback = feedback_data.count(0)
        #plotting
        fig, ax = plt.subplots()
        ax.bar(['Positive', 'Negative'], [positive_feedback, negative_feedback])
        st.pyplot(fig)

        #Comparing the 'feedback' column with 'rating' column
        st.write('Comparing feedback with rating')
        st.write(data[data['feedback']==0]['rating'].value_counts())
        st.write(data[data['feedback']==1]['rating'].value_counts())
        st.write('1 and 2 stars indicate negative feedbacks while')
        st.write('3, 4 and 5 stars indicate positive feedbacks.')

    #Analyzing the 'variation' column
    if st.checkbox('Show Variation Analysis'):
        st.write('Variation Analysis')
        st.write(data.groupby('variation').describe())

        st.write('Most Popular Variations')
        st.write(data['variation'].value_counts())

        #Plotting the mean variation ratings
        st.write('Mean Variation Ratings')
        st.write(data.groupby('variation')['rating'].mean())
        fig, ax = plt.subplots()
        data.groupby('variation')['rating'].mean().plot(kind='bar', ax=ax)
        st.pyplot(fig)
        
    #Finding length of each review
    if st.checkbox('Length of Reviews'):
        data['length'] = data['verified_reviews'].apply(len)
        #Plotting histogram
        st.write('Histogram of Review Length')
        fig, ax = plt.subplots()
        data['length'].plot(kind='hist', bins=50, ax=ax)
        st.pyplot(fig)
        st.write("Length of reviews 0")
        fig, ax =plt.subplots()
        data[data['feedback']==0]['length'].plot(kind ='hist', ax=ax )
        st.pyplot(fig)
        st.write("Length of reviews 1")
        fig, ax =plt.subplots()
        data[data['feedback']==1]['length'].plot(kind ='hist', ax=ax )
        st.pyplot(fig)
        
    # Feature Extraction
    if st.checkbox("Extract Feature"):
        cv = CountVectorizer(stop_words='english')
        words= cv.fit_transform(data.verified_reviews)
        
        pol =lambda x: TextBlob(x).sentiment.polarity
        sub= lambda x: TextBlob(x).sentiment.subjectivity
        data['polarity']=data['verified_reviews'].apply(pol)
        data['subjectivity']= data['verified_reviews'].apply(sub)
        st.write(data)
        
        #Plot based on polarity score
        st.write("Histogram of polarity score")
        num_bins=50
        plt.figure(figsize=(10,6))
        fig,ax=plt.subplots()
        data['polarity'].plot(kind = 'hist',bins = num_bins, ax =ax)
        st.pyplot(fig)
        
        st.write("Histogram of Subjectivity score")
        plt.figure(figsize=(10,6))
        fig,ax=plt.subplots()
        data['subjectivity'].plot(kind = 'hist',bins = num_bins, ax =ax)
        st.pyplot(fig)
    
    #Now we try to use some NLP tools for the data pre-processing. 
    #We are using PorterStemmer algorithm to remove suffixes from words.
    if st.checkbox(" Process Text"):
        nltk.download('stopwords')
        stopwords=set(stopwords.words('english'))
        corpus = []
        stemmer = PorterStemmer()
        for i in range(0,data.shape[0]):
            review = re.sub('[^a-zA-Z]',' ',data.iloc[i]['verified_reviews'])
            review = review.lower().split()
            review = [stemmer.stem(word) for word in review if not word in stopwords]
            review = ' '.join(review)
            corpus.append(review)
        
        cv =CountVectorizer(max_features=2500)
        x=cv.fit_transform(corpus).toarray()
        y=data['feedback'].values
        
        
        pickle.dump(cv,open('countVectorizer.pkl', 'wb'))
        
        st.write("Text processed Succesfully")
        
    if st.checkbox("Train and Test model"):
        X_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=15)
        model_rf = RandomForestClassifier()
        scaler = MinMaxScaler()
        X_train_scl = scaler.fit_transform(X_train)
        x_test_scl = scaler.transform(x_test)
        model_rf.fit(X_train_scl,y_train)
        RandomForestClassifier()
        st.write("Training Accuracy")
        st.write(model_rf.score(X_train_scl,y_train))
        st.write("Testing accuracy")
        st.write(model_rf.score(x_test_scl, y_test))
        
        y_preds = model_rf.predict(x_test_scl)
    
    if st.checkbox("Show confusion Matrix"):
        cm = confusion_matrix(y_test, y_preds)
        cm_display=ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model_rf.classes_)
        cm_display.plot()
        plt.title('Confusion Matrix')
        plt.show()
    
    if st.checkbox("Test using XGBOOST"):
        xgb_classifier = xgb.XGBClassifier()
        xgb_classifier.fit(X_train, y_train)
        
        y_pred = xgb_classifier.predict(x_test)
        st.write("Accuracy")
        st.write(accuracy_score(y_test, y_pred))
         
    

    #Downloading the processed dataset
# Check if the user clicked the "Download Processed Data" button
    if st.button('Download Processed Data'):
        # Convert the DataFrame to a CSV file
        csv = data.to_csv(index=False)
        # Create a link for the user to click and download the file
        st.download_button(
            label='Download CSV',
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv'
        )

if st.button("About Application"):
    st.text('It is developed by Ayush Das, Sayan Banerjee, Sreejata Banerjee,')
    st.text('Lucky Das and Mrinalini Bhattacharjee')
    

