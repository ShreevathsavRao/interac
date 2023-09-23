from flask import Flask, request, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime
import snowflake.connector as sf
import numpy as np
from GoogleNews import GoogleNews
from datetime import timedelta
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('all')
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()
import calendar
import time

app = Flask(__name__)

def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    stem_words=[stemmer.stem(w) for w in filtered_words]
    lemma_words=[lemmatizer.lemmatize(w) for w in stem_words]
    return " ".join(filtered_words)

def get_news(time_stamp,title,gap = 60):
    stop_words = set(stopwords.words('english'))
    processed=[]
    line = title
    words = line.split()
    for r in words:
        if not r in stop_words:
            processed.append(r)
    string = " ".join(processed)
    string = preprocess(string)
    time_stamp = datetime.strptime(str(time_stamp), '%Y-%m-%d %H:%M:%S')
    yesterday = time_stamp - timedelta(minutes=gap)
    googlenews=GoogleNews(start=yesterday.strftime("%m-%d-%Y"),end=time_stamp.strftime("%m-%d-%Y"))
    googlenews.get_news(string)
    result=googlenews.result()
    df=pd.DataFrame(result)
    return df

def views_prediction2(string):
    # create a connection object
    conn = sf.connect(
        user='ankan',
        password='ankanProboReset@123',
        account='rl48423.ap-south-1.aws',
        database='MELTANO_DB',
        schema='PROBO_ANALYTICS'
    )
    cursor = conn.cursor()
    cursor.execute(string)
    data = cursor.fetchall()
    df = pd.DataFrame(data, columns=[desc[0] for desc in cursor.description])
    return df

def process_input_value(input_value):
    event_price_time_qstring = 'select BUY_price,CREATED_DT from probo_analytics.tms_trade WHERE event_id={}'
    event_name_desc_qstring = 'select NAMe,DESCRIPTION from probo_analytics.EVENTS WHERE ID={}'
    print(input_value)
    event_price_time = views_prediction2(event_price_time_qstring.format(input_value))
    event_name_desc = views_prediction2(event_name_desc_qstring.format(input_value))
    print('Event name:',event_name_desc['NAME'][0])
    event_price_time = event_price_time.sort_values('CREATED_DT')
    event_price_time['rate_change'] = event_price_time['BUY_PRICE'].pct_change(fill_method='ffill')
    event_price_time['get_data_change'] = np.where(abs(event_price_time['rate_change'])>=event_price_time['rate_change'].std() ,event_price_time['BUY_PRICE'],None)    
    event_price_time['get_data'] = np.where(event_price_time['get_data_change'] ,True,False)
    event_price_time.reset_index(drop=True,inplace=True)

    time_stamp = []
    time_stamp = event_price_time.query('get_data == True')['CREATED_DT']
    print('Total Signals:',len(time_stamp))
    title = event_name_desc['NAME'][0]
    # description = event_name_desc['DESCRIPTION'][0]

    end_abbr_month = datetime.strptime(str(max(event_price_time['CREATED_DT'])),'%Y-%m-%d %H:%M:%S').month
    str_abbr_month = datetime.strptime(str(min(event_price_time['CREATED_DT'])),'%Y-%m-%d %H:%M:%S').month

    months = []
    for i in range(str_abbr_month,end_abbr_month+1):
        months.append(calendar.month_abbr[i])
    count = 0
    news_group = {}
    for i in range(int(len(event_price_time['BUY_PRICE']))):
            if event_price_time['get_data'][i] == True and i<22485:
                fun_start_time = datetime.now()
                count+=1
                news = get_news(event_price_time['CREATED_DT'][i] ,title)
                try:
                    for k in range(len(news)):
                        for ele in months:
                            if news['title'][k] not in news_group and  ele in news['date'][k]:
                                news_group_sub = {}
                                news_group_sub['title'] = news['title'][k]
                                news_group_sub['datetime'] = news['datetime'][k]
                                news_group_sub['desc'] = news['desc'][k]
                                news_group_sub['date'] = news['date'][k]
                                news_group_sub['link'] = news['link'][k]
                                news_group_sub['img'] = news['img'][k]
                                news_group_sub['media'] = news['media'][k]
                                news_group_sub['site'] = news['site'][k]
                                news_group_sub['sentiment'] = (sentiment.polarity_scores(news['title'][k]))['compound']
                                news_group[(time.mktime(datetime.strptime(str( news['datetime'][k]), '%Y-%m-%d %H:%M:%S').timetuple())+0)*1000] = news_group_sub
                except:
                    pass
                
                print('Finished {} : {} : {}/{} : {}'.format(i,input_value,count,len(time_stamp), datetime.now()-fun_start_time))

    end_date = time.mktime((datetime.strptime(str(max(event_price_time['CREATED_DT'])).split('.')[0], '%Y-%m-%d %H:%M:%S')).timetuple())*1000
    print('end_date:',end_date)
    start_date= time.mktime((datetime.strptime(str(min(event_price_time['CREATED_DT'])).split('.')[0], '%Y-%m-%d %H:%M:%S')).timetuple())*1000
    print('start_date:',start_date)

    event_gap = (end_date-start_date)/20
    second_time = start_date
    filtered_data = {}
    group=0
    while second_time<end_date:
        first_time=second_time
        second_time = second_time+event_gap
        filtered_data[group]=[]
        for i in news_group:
            if  int(first_time) <= i <= int(second_time):
                filtered_data[group].append( news_group[i])
        group=group+1
    # return json.dumps(news_group.to_dict(), indent = 4) 
    print('\n Total number of newses:',len(filtered_data),'\n')
    return filtered_data

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_input():
    input_value = request.json.get('input_value')
    if input_value is not None:
        output = process_input_value(input_value)
        return jsonify(data=output)
    return jsonify(error='Input value not provided'), 400

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0',port=8080)  # Change 8080 to the desired port number
