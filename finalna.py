import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math
from decimal import Decimal
from datetime import datetime, timedelta
import tweepy
import re
from nltk.tokenize import TweetTokenizer
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import advertools as adv
from nltk.stem.snowball import SnowballStemmer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from wordcloud import WordCloud
import base64
import pickle
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

# Utils
import base64 
import time
timestr = time.strftime("%Y%m%d-%H%M%S")

st.set_page_config(page_title='Sentiment Classifier Tool', page_icon="🐦")

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
      
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.set_option('deprecation.showPyplotGlobalUse', False)
sns.set()

def csv_downloader(data, name):
	csvfile = data.to_csv()
	b64 = base64.b64encode(csvfile.encode()).decode()
	new_filename = name + ".csv"
	st.markdown("#### Download File ###")
	href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Click Here!!</a>'
	st.markdown(href,unsafe_allow_html=True)

def download_model(model):
  output_model = pickle.dumps(model)
  b64 = base64.b64encode(output_model).decode()
  st.markdown("#### Download File ###")
  href = f'<a href="data:file/output_model;base64,{b64}" download="{"Model.pkl"}">Download Trained Model .pkl File</a>'
  st.markdown(href, unsafe_allow_html=True)


st.markdown("Tweets Evaluation Wizard")
steps = st.select_slider("Select Stage",["1.Collect","2.Clean","3.Train","4.Classify","5.Visualize"])

if steps == "1.Collect":
  st.title('Tweets Collection')
  source = st.sidebar.selectbox("Enter Source of Tweets",['Online',"File"])
  if source == "Online":
    query = st.sidebar.text_input('Query', 'Enter your twitter query')
    limit = st.sidebar.slider('Limit',10,1000,100)
    #language = st.sidebar.selectbox("Please Select Language of Tweets",['English',"Filipino"])
    start = st.sidebar.date_input('Start Date', datetime.now() - timedelta(days=5),datetime.now() - timedelta(days=7),datetime.now())
    end = st.sidebar.date_input('End Date', datetime.now(),start,datetime.now())
    query = query + " -is:retweet"
    start = str(start) + " 00:00:00"
    end = str(end) + " 00:00:00"
    start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end =   datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    st.session_state.query = query
    st.session_state.limit = limit
    st.session_state.start = start
    st.session_state.end = end
    st.session_state.ApiKey = "muIy9n6QlYjcbQaAfZdUaGJti"
    st.session_state.ApiKeySecret = "Sz54GsqX21wVofh8d6g8KiSgXE7TEXrYrbRJv0aNOgaJSkIMEg"
    st.session_state.AccessToken = "722832546707320835-xCAgkJtvRQb9U4edUhODtbFTF9b0k6P"
    st.session_state.AccessTokenSecret = "6K0WYY5nAB3LTxH8k1dpny05T6tQjhUbOxzORHUbEFhK5"
    st.session_state.BearersToken = "AAAAAAAAAAAAAAAAAAAAAKPbOQEAAAAAhkEfzKaoBDoR%2BZfHzTKEr7F5CCo%3DOTdI3WElJ5ctnpr189sEe9u0jHsMJY102riRgnUAtX3FnvFt9R"
    fetch = st.sidebar.button("Fetch Tweets")
    if fetch == True:
      client = tweepy.Client(bearer_token=st.session_state.BearersToken, consumer_key=st.session_state.ApiKey, consumer_secret=st.session_state.ApiKeySecret, 
                        access_token=st.session_state.AccessToken, access_token_secret=st.session_state.AccessTokenSecret, 
                        wait_on_rate_limit=True)
      tweets = client.search_recent_tweets(query=query,tweet_fields = ['created_at','text',"lang"], start_time =start ,end_time =end,
                                      max_results=max(10,min(100,limit)))
      print(tweets.meta)
      next_token = tweets.meta['next_token']
      twts = []
      for tweet in tweets.data:
        #if language == "English":
        if tweet['lang'] == 'en':
          twts.append(tweet)
        #elif language == "Filipino":
        elif tweet['lang'] == 'tl':
          twts.append(tweet)
      while len(twts) < limit:
        tweets = client.search_recent_tweets(query=query,tweet_fields = ['created_at','text',"lang"], start_time =start ,end_time =end,
                                            max_results=max(10,min(100,limit-len(twts))),next_token = next_token)
        try:
          next_token = tweets.meta['next_token']
        except KeyError:
          print(tweets.meta)
          break
        for tweet in tweets.data:
          #if language == "English":
          if tweet['lang'] == 'en':
            twts.append(tweet)
          #elif language == "Filipino":
          elif tweet['lang'] == 'tl':
            twts.append(tweet)
        print("Next Search")
      twts = twts[0:limit]
      st.session_state.tweets = twts
      st.write(st.session_state.tweets)
      twts_df = pd.DataFrame(st.session_state.tweets)
      csv_downloader(twts_df,"fetched")
      
  elif source == "File":
    uploadFile = st.sidebar.file_uploader("Choose a Tweets file","json")
    if uploadFile is not None:
      #language = st.sidebar.selectbox("Please Select Language of Tweets",['English',"Filipino"])
      fileTweets = uploadFile.read()
      fileTweets = fileTweets.split(b'\n')
      data = []
      k = 0
      l = 0
      for index,line in enumerate(fileTweets):
        k = k+1
        try:
          json_object = json.loads(line)
          dte = json_object['date'] + " 00:00:00"
          dte = datetime.strptime(dte, '%Y-%m-%d %H:%M:%S')
          sel_json = {"created_at": dte,
                      "text": json_object['tweet'],
                      "lang":json_object['language']}
          data.append(sel_json)
        except Exception as e:
          l = l +1
          print("Error: "+ str(l))
          print(e)
      twts = []
      for tweet in data:
        #if language == "English":
        if tweet['lang'] == 'en':
          twts.append(tweet)
        #elif language == "Filipino":
        elif tweet['lang'] == 'tl':
          twts.append(tweet)
      st.session_state.tweets = twts
      st.write(st.session_state.tweets)
      twts_df = pd.DataFrame(st.session_state.tweets)      
      csv_downloader(twts_df,"fetched")
      



if steps == "2.Clean":
  st.title('Tweets Cleaning')
  df_tweets = pd.DataFrame(st.session_state.tweets)
  st.sidebar.title("Preprocessing")
  lower = st.sidebar.checkbox('Lower Case')
  no_punct = st.sidebar.checkbox('Remove Punctuation')
  no_url = st.sidebar.checkbox('Remove URLs')
  no_num = st.sidebar.checkbox('No Numbers')
  no_handles = st.sidebar.checkbox('Remove Handles')
  token = st.sidebar.checkbox('Tokenization')
  no_sw = st.sidebar.checkbox('No Stopwords (Requires Tokenization)')
  if no_sw == True:
    add_stopwords = st.sidebar.text_area("Enter additional stopwords seperated by comma")
    stopwords_english = stopwords.words('english')
    stopwords_filipino = list(adv.stopwords['tagalog'])
    additional_filipino = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay",
                          "bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin",
                          "gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag",
                          "iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas",
                          "ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino",
                          "kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad", "kaya", "kaysa", "ko", "kong",
                          "kulang", "kumuha", "kung", "laban", "lahat", "lamang","likod","lima","maaari","maaaring","maging",
                          "mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na",
                          "nabanggit", "naging",  "nagkaroon",  "nais", "nakita", "namin", "napaka", "narito", "nasaan", "ng", "ngayon",
                          "ni", "nila", "nilang", "nito", "niya", "niyang", "noon", "o","pa","paano","pababa","paggawa","pagitan",
                          "pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas",
                          "pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino", "siya", "tatlo", "tayo",
                          "tulad", "tungkol", "una"]
    Hashtag_keywords = ["#ping", "#halalan", "#isko" , "#leniforpresident", "#iskomoreno", "#lacson-sotto", "#pinglacson", "#mannypacquiao", "#bbm", "#lenirobredo", "#bbmismypresident", "#pinglacson",  
                        "#bbmsara", "#bangonbayanmuli","#iampinglacson", "#iskonaman", "#kayiskoposible","#iskoparasapilipino","#biliskilos","#iskoparasabayan","#lacson", "#bbmsarauniteam", 
                        "#isko2022","#ikawangnaisko","#naisko2022", "#iskomorenoForpresident", "#withdrawisko", "#iskoourpresident","#labanleni2022","#letlenilead",  "#iampinglacson", 
                        "#lenimatapang","#lenikiko2022","#lenibusypresidente","#leniangatsalahat", "#leniduwag","#robreduwag","#letlenilive","#leni","#presidentlenirobredo", "#lacsonsotto",
                        "#lenirobredo2022","#leniforpresident2022","#bbm","#bbm2022","#wechoosemarcos", "#labanleni","#marcosduwagparin","#bbmsara2022","#bbmsarauniteam2022", "#bongbongmarcos",
                        "#marcosparin","#bbmismypresident2022","#bbmforpresident2022","#neveragain","#marcosmagnanakaw", "#bbmsigawngbayan","#bbm2022tomakephilippinesgreatagain", "#eleksyon",
                        "#solidbbm","#kakamping","#pinglacson2022", "#lenikiko","#weneedaleader","#pinggaling","#ping17thpresident", "#kulayrosasangbukas" "#lacsonsotto2022","#lacsonforpresident",
                        "#aayusinanggobyerno","#kulayrosasangbukas", "#pingmostqualifiedpresident","#kaypingtayo","#pacquiao2022","#pacquiaoforpresident","#mptayo", "#panaloangmahirap",
                        "#mannypacquiao2022","#forgodandcountry","#mppm","#kakampink" ,"#kakampinks","#angatbuhaylahat","#voteph2022", "#teamleni","#kapinkbisig","#kaylenitayoangpanalo",
                        "#leniwalangaatrasan","#siguradotayokayleni","#bangonbayanmuli","#teambongbong","#uniteambbmsara", "#pulaangkulayngrosas","#orasnamarcosna","#leadershipbyexample",
                        "#lacsonsottotayo", "#halalan2022", "#eleksyon2022", "#mahalinnatinangpilipinas", "#uniteambbmsara", "#mannypacquiao","#pacquiao","#manny", "#mannypacquiaoforpresident"]

    unHashtag_keywords = ["isko" , "ping",  "iskomoreno",  "lacson-sotto", "moreno", "mannypacquiao", "bbm", "lenirobredo", "bbmismypresident", "bongbong", "marcos",  "bongbongmarcos", 
                        "bbmsara", "mannypacquiao", "halalan", "pacquiao","manny", "eleksyon", "leni", "robredo", "lacson", "si", "pinglacson", "iampinglacson", "lacsonsotto"]
    given_sw = add_stopwords.split(",")
    st.session_state.given_sw = given_sw
    all_stopwords = stopwords_english + stopwords_filipino + additional_filipino + Hashtag_keywords + unHashtag_keywords
    selected_stopwords = st.sidebar.multiselect("Remove Stopwords",all_stopwords,all_stopwords)
    selected_stopwords = selected_stopwords + given_sw
  #stemmed = st.sidebar.checkbox('Stemming (Requires Tokenization)')
  preprocess = st.sidebar.button("Start Preprocessing")
  
  st.session_state.lower = lower
  st.session_state.no_punct = no_punct
  st.session_state.no_url = no_url
  st.session_state.no_num = no_num
  st.session_state.no_handles = no_handles
  st.session_state.token = token
  st.session_state.no_sw = no_sw
  #st.session_state.stemmed = stemmed
  st.session_state.preprocess = preprocess
  

  if preprocess == True:
    df_tweets['day'] = df_tweets['created_at'].dt.day_name()
    df_tweets['clean'] = df_tweets['text']
    if lower == True:
      df_tweets['clean'] = df_tweets['clean'].str.lower()
      df_tweets['clarity'] = df_tweets['clean']
    if no_punct == True:
      df_tweets['clean'] = df_tweets["clean"].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
      df_tweets['clarity'] = df_tweets['clean']
    if no_url == True:
      df_tweets['clean'] = df_tweets["clean"].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
      df_tweets['clarity'] = df_tweets['clean']
    if no_num == True:
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
      df_tweets['clarity'] = df_tweets['clean']
    if no_handles == True:
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: re.sub(r'@mention', '', x))
      df_tweets['clarity'] = df_tweets['clean']
    if token == True:
      tknzr = TweetTokenizer()
      df_tweets['clean'] = df_tweets['clean'].apply(tknzr.tokenize)
    if no_sw == True:
      
      st.session_state.selected_stopwords = selected_stopwords 
      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in stopwords_english]
      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in stopwords_filipino]
      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in additional_filipino]
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in stopwords_english])
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in stopwords_filipino])
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in additional_filipino])
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in selected_stopwords])


    #if stemmed == True:
    #  stemmer = SnowballStemmer("english")
    #  df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [stemmer.stem(y) for y in x])
    df_tweets = df_tweets[df_tweets["clean"].str.len() != 0]
    st.session_state.preprocessed = df_tweets
    df_display = df_tweets[['lang','text','clean']]
    st.table(df_display)
    csv_downloader(df_tweets,"cleaned")

if steps == "3.Train":
  st.title('Training Tweets Model')
  st.sidebar.title("Training Models")
  trainingFile = st.sidebar.file_uploader("Choose a training file","json")
  algorithm = st.sidebar.selectbox("Select the Algorithm",["Naive Bayes",'SVM'])
  data = []
  k = 0
  l = 0
  if trainingFile is not None:
    fileTrain = trainingFile.read()
    fileTrain = fileTrain.split(b'\n')
    data = []
    k = 0
    l = 0
    for index,line in enumerate(fileTrain):
      k = k+1
      try:
        json_object = json.loads(line)
        data.append(json_object)
      except Exception as e:
        l = l +1
    dfTrain = pd.DataFrame(data)
    dfTrain = dfTrain.loc[(dfTrain['label'] == -1) | (dfTrain['label'] == 0) | (dfTrain['label'] == 1)]
    df_tweets = dfTrain.copy()
    df_tweets['clean'] = df_tweets['tweet']
    if st.session_state.lower == True:
      df_tweets['clean'] = df_tweets['clean'].str.lower()
      
    if st.session_state.no_punct == True:
      df_tweets['clean'] = df_tweets['clean'].astype(str)
      df_tweets['clean'] = df_tweets["clean"].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
      
    if st.session_state.no_url == True:
      df_tweets['clean'] = df_tweets['clean'].astype(str)
      df_tweets['clean'] = df_tweets["clean"].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
      
    if st.session_state.no_num == True:
      df_tweets['clean'] = df_tweets['clean'].astype(str)
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
      
    if st.session_state.no_handles == True:
      df_tweets['clean'] = df_tweets['clean'].astype(str)
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: re.sub(r'@mention', '', x))
      
    if st.session_state.token == True:
      tknzr = TweetTokenizer()
      df_tweets['clean'] = df_tweets['clean'].apply(tknzr.tokenize)
    if st.session_state.no_sw == True:
      stopwords_english = stopwords.words('english')
      stopwords_filipino = list(adv.stopwords['tagalog'])
      additional_filipino = ["akin","aking","ako","alin","am","amin","aming","ang","ano","anumang","apat","at","atin","ating","ay",
                             "bababa","bago","bakit","bawat","bilang","dahil","dalawa","dapat","din","dito","doon","gagawin",
                             "gayunman","ginagawa","ginawa","ginawang","gumawa","gusto","habang","hanggang","hindi","huwag",
                             "iba","ibaba","ibabaw","ibig","ikaw","ilagay","ilalim","ilan","inyong","isa","isang","itaas",
                             "ito","iyo","iyon","iyong","ka","kahit","kailangan","kailanman","kami","kanila","kanilang","kanino",
                             "kanya","kanyang","kapag","kapwa","karamihan","katiyakan","katulad", "kaya", "kaysa", "ko", "kong",
                            "kulang", "kumuha", "kung", "laban", "lahat", "lamang","likod","lima","maaari","maaaring","maging",
                             "mahusay","makita","marami","marapat","masyado","may","mayroon","mga","minsan","mismo","mula","muli","na",
                             "nabanggit", "naging",  "nagkaroon",  "nais", "nakita", "namin", "napaka", "narito", "nasaan", "ng", "ngayon",
                             "ni", "nila", "nilang", "nito", "niya", "niyang", "noon", "o","pa","paano","pababa","paggawa","pagitan",
                             "pagkakaroon","pagkatapos","palabas","pamamagitan","panahon","pangalawa","para","paraan","pareho","pataas",
                             "pero","pumunta","pumupunta","sa","saan","sabi","sabihin","sarili","sila","sino", "siya", "tatlo", "tayo",
                              "tulad", "tungkol", "una"]
      given_sw = st.session_state.given_sw
      #all_stopwords = stopwords_english + stopwords_filipino + additional_filipino + given_sw
      selected_stopwords = st.session_state.selected_stopwords

      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in stopwords_english]
      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in stopwords_filipino]
      #df_tweets['clean'] = [word for word in df_tweets['clean'] if not word in additional_filipino]
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in stopwords_english])
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in stopwords_filipino])
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in additional_filipino])
      #df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in given_sw])
      df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [item for item in x if item not in selected_stopwords])
    #if st.session_state.stemmed == True:
    #  stemmer = SnowballStemmer("english")
    #  df_tweets['clean'] = df_tweets['clean'].apply(lambda x: [stemmer.stem(y) for y in x])
    df_train = df_tweets[['clean',"label"]]
    df_train = df_train.dropna()
    s = df_train['clean']
    mlb_fin = MultiLabelBinarizer()
    df_dummy_fil = pd.DataFrame(mlb_fin.fit_transform(s),columns=mlb_fin.classes_, index=df_train.index)
    df_dummy_fil['target_labels']=df_train['label']
    X = df_dummy_fil.copy()
    X = X.drop("target_labels",axis = 1)
    y = df_dummy_fil['target_labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state = 42)
    if algorithm == "Naive Bayes":
      clf = MultinomialNB()
    elif algorithm == "SVM":
      clf = LinearSVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    cm = metrics.confusion_matrix(y_test,y_pred)
    fig = px.imshow(cm, title = "Confusion Matrix",labels=dict(x="Predicted", y="Actual", color="Productivity"),
                x=["Negative","Neutral","Positive"],
                y=["Negative","Neutral","Positive"]
               )
    CR = metrics.classification_report(y_test,y_pred,target_names = ['Negative',"Neutral","Positive"], digits=4)
    st.text('Model Report:\n ' + CR)
    st.plotly_chart(fig)
    st.session_state.algorithm = algorithm
    st.session_state.clf = clf
    st.session_state.features = list(X_train.columns)
    download_model(clf)
 
if steps == "4.Classify":
  process = st.sidebar.selectbox('Do you want to input model and test data',['yes','no'])
  if process == "yes":
    model =  st.sidebar.file_uploader("Choose a training file","pkl")
    test_data = st.sidebar.file_uploader("Choose Test Data","csv")
    if (model is not None) & (test_data is not None):
      st.title('Tweets Classifications by')
      st.title("Selected Model")  
      df_new = pd.read_csv(test_data)
      with open(model.name,"wb") as fl:
        fl.write(model.getbuffer())      
      clf = pickle.load(open(model.name, 'rb'))
      features = st.session_state.features
      df_new['clean'] = df_new['text']
      
      if st.session_state.lower == True:
        df_new['clean'] = df_new['clean'].str.lower()
      
      if st.session_state.no_punct == True:
        df_new['clean'] = df_new['clean'].astype(str)
        df_new['clean'] = df_new["clean"].apply(lambda x: re.sub(r'https?:\/\/\S+', '', x))
      
      if st.session_state.no_url == True:
        df_new['clean'] = df_new['clean'].astype(str)
        df_new['clean'] = df_new["clean"].apply(lambda x: re.sub(r"www\.[a-z]?\.?(com)+|[a-z]+\.(com)", '', x))
      
      if st.session_state.no_num == True:
        df_new['clean'] = df_new['clean'].astype(str)
        df_new['clean'] = df_new['clean'].apply(lambda x: re.sub(r"[^a-z\s\(\-:\)\\\/\];='#]", '', x))
        
      if st.session_state.no_handles == True:
        df_new['clean'] = df_new['clean'].astype(str)
        df_new['clean'] = df_new['clean'].apply(lambda x: re.sub(r'@mention', '', x))
        
      if st.session_state.token == True:
        tknzr = TweetTokenizer()
        df_new['clean'] = df_new['clean'].apply(tknzr.tokenize)
      if st.session_state.no_sw == True:
        selected_stopwords = st.session_state.selected_stopwords
      df_new['clean'] = df_new['clean'].apply(lambda x: [item for item in x if item not in selected_stopwords])
      
      X_new = pd.DataFrame(0, index=np.arange(len(df_new)), columns=features)
      X_new['clean_text'] = df_new['clean']
      X_new["clean_text"] = [x for x in X_new['clean_text']]
      for index, row in X_new.iterrows():
        keywords = row['clean_text']
        if isinstance(keywords, list):
          for k in keywords:
            if k in features:
              X_new.loc[index,k] = 1
      X_new = X_new.drop(['clean_text'],axis=1)
      results = clf.predict(X_new)
      df_new['sentiments'] = results
      st.session_state.results = df_new
      df_new["sentiment"] = "Neutral"
      df_new.loc[df_new['sentiments'] == 1, "sentiment"] =  "Positive"
      df_new.loc[df_new['sentiments'] == -1, 'sentiment'] = "Negative"
      df_display = df_new[['lang','text','clean','sentiment']]
      st.write(df_display)
      df_dis = pd.DataFrame(df_display)
      csv_downloader(df_new,"classified")
  
  elif process == "no":    
    st.title('Tweets Classifications by')
    st.title(st.session_state.algorithm)  
    df_new = st.session_state.preprocessed
    features = st.session_state.features
    X_new = pd.DataFrame(0, index=np.arange(len(df_new)), columns=features)
    X_new['clean_text'] = df_new['clean']
    X_new["clean_text"] = [x for x in X_new['clean_text']]
    for index, row in X_new.iterrows():
      keywords = row['clean_text']
      if isinstance(keywords, list):
        for k in keywords:
          if k in features:
            X_new.loc[index,k] = 1
    X_new = X_new.drop(['clean_text'],axis=1)
    clf = st.session_state.clf
    results = clf.predict(X_new)
    df_new['sentiments'] = results
    st.session_state.results = df_new
    df_new["sentiment"] = "Neutral"
    df_new.loc[df_new['sentiments'] == 1, "sentiment"] =  "Positive"
    df_new.loc[df_new['sentiments'] == -1, 'sentiment'] = "Negative"
    df_display = df_new[['lang','text','clean','sentiment']]
    st.write(df_display)  
    csv_downloader(df_new,"classified")


if steps == "5.Visualize":
  st.title('Tweets Visualizations')
  results = st.session_state.results
  chart = st.sidebar.radio("Select Visualization",["Classification Chart",'Word Cloud',"Word Frequency Plot"])
  results['joined'] = results['clean'].apply(' '.join)
  if chart == "Classification Chart":
    my_layout = px.histogram(results, x='created_at', color= "sentiment", barmode = "group", labels={"created_at": "DATES", "count": "Number of Sentiments", "sentiments" : "Sentiment"
                 },title="Sentiments count October to February") 
    
    st.plotly_chart(my_layout) 
  elif chart == "Word Cloud":
    text = results['joined'].values
    wordcloud = WordCloud().generate(str(text))
    fig = plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    st.pyplot()
  #elif chart == "Sentiment Analysis":
   # fig = px.histogram(results, x='sentiments')
    #st.plotly_chart(fig, use_container_width=True)
  elif chart == "Word Frequency Plot":
    counts = results['joined'].str.findall(r"(\w+)").explode().value_counts()
    top_10 = counts.nlargest(10)
    fig = px.bar(top_10, labels={"index": "Words",
                   "value": "Word count",
                 },title="Word Frequencies") 
    fig.layout.update(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
