import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys
import urllib.request
import urllib
import csv
import os
import zipfile
from zipfile import ZipFile
from urllib.request import urlopen
from io import BytesIO
import glob
import numpy as np

#import urllib2

year = input("Enter the year ")
baseURL = "https://www.sec.gov/"
url = baseURL + "files/edgar"+ year +".html"
print(url)
link = []
soup = BeautifulSoup((urllib.request.urlopen(url)),"html.parser")
list_links = soup.select('ul li')
for list_link in list_links:
	for li in list_link.find_all('a'):
		log_string = str(li.text).split('.')[0]
		if ("01" in log_string[9:11]):
			href = li['href']	
			link.append(href)
			

foldername = str(year)
path = str(os.getcwd()) + "/" + foldername

for file in link:
	print(file)
	filename = file.split('/')[-1]
	filename1 = filename.split('.')[0]
	with urlopen(file) as zipresp:
		with ZipFile(BytesIO(zipresp.read())) as zfile:
			zfile.extractall(path)


allFiles = glob.glob( path + "/log*.csv")
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_ ,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)

'''
	cs1 = csv1.groupby(pd.cut(csv1['size'], np.percentile(csv1['size'], [0, 25, 75, 90, 100]))).count()
	csv1Plot = csv1.groupby('time')['cik'].count()
	def convert_totime(seconds):
	return datetime.datetime.utcfromtimestamp(seconds);

	timeAnalysis = csv1[['time' , 'cik']].copy()

	timeAnalysis['datetime'] = timeAnalysis.time.apply(convert_totime)
	# As the max time is 172792 seconds and 172792 / (60*60) is about 48 hrs so we only have data for 2 days so only 
	# plotting data against hours make sense
	timeAnalysis['hour of the day'] = timeAnalysis.datetime.dt.hour
	timeAnalysisGrouped = timeAnalysis('hour of the day')['cik'].count()
	cs1 = cs1.rename(columns={'size':'old_size'})
	cs1['size'].value_counts().plot(kind= 'bar',figsize = (50,20), fontsize = 50)
	csv1['code'].value_counts().plot(kind= 'bar',figsize = (50,20), fontsize = 50)
	plt.xlabel('Status Code',fontsize = 50)
	plt.ylabel('No of filings', fontsize = 50)
	csv1.groupby(['ip','cik', 'accession' ]).count()
	pass
'''










def replacing_missing_values(csv1):
	new_data = pd.DataFrame()
	csv1['ip'].fillna(value= "000.00.00.000" , inplace = True)
	csv1['date'].fillna(method = 'bfill' , inplace = True)
	csv1['time'].fillna(method = 'ffill' , inplace = True)
	csv1['zone'].fillna(method = 'ffill' , inplace = True) 
	csv1 = csv1.reset_index(drop = True)
	#csv1.drop('index', axis = 1, inplace=True)
	index = csv1.index[csv1["extention"] == ".txt"]
	csv1.set_value(index, "extention", (csv1['accession'].map(str) + csv1['extention']))
	csv1 = csv1.set_index("extention")
	#csv1["extention"] = csv1.index
	#as_list = csv1.extention.tolist()
	#idx = as_list.index(".txt")
	#as_list[idx] = (csv1['accession'].map(str) + csv1['extention'])
	#csv1.extention = as_list 
	#csv1.loc[csv1['extention'] == '.txt' , "extention"].reset_index() 
	#csv1.set_value(index, "extention", (csv1['accession'].map(str) + csv1['extention']))
	#csv1 = csv1.set_index("extention")

	#csv1['extention']  = 
	#csv1.reset_index(csv1['extention'])
	csv1['code'].fillna(value= 0 , inplace = True)
	csv1['size'] = csv1['size'].fillna(value= 0)
	csv1['size'] = csv1['size'].astype('int64')
	csv1['idx'].fillna(value= 0 , inplace = True)
	csv1['norefer'].fillna(value= 0 , inplace = True)
	csv1['noagent'].fillna(value= 0 , inplace = True)
	csv1['find'].fillna(value= 0 , inplace = True)
	#csv1['crawler'][(csv1['noagent'] is '0') or (csv1['code'] is '404')] = 1
	#csv1.loc[csv1['noagent'] == 1,"crawler"] = csv1['crawler'].replace(np.nan,0)
	default = "win"
	csv1['browser'] = csv1['browser'].replace(np.nan,default)
	#csv1.loc[csv1['noagent'] == 0,"browser"] = csv1['browser'].replace(np.nan,default)
	#csv1.loc[csv1['noagent'] == 1,"browser"] = csv1['browser'].replace(np.nan,"not defined")
	new_data = csv1
	return new_data



def changing_datatypes(csvdata):
	#if(csvdata.values() != ''):
	csvdata['date'] = pd.to_datetime(csvdata['date'])
	csvdata['zone'] = csvdata['zone'].astype('int64')
	csvdata['cik'] = csvdata['cik'].astype('int64')
	csvdata['code'] = csvdata['code'].astype('int64')
##	csvdata['size'] = csvdata['size'].astype('int64')
	csvdata['idx'] = csvdata['idx'].astype('int64')
	csvdata['norefer'] = csvdata['norefer'].astype('int64')
	csvdata['noagent'] = csvdata['noagent'].astype('int64')
	csvdata['find'] = csvdata['find'].astype('int64')
	csvdata['crawler'] = csvdata['crawler'].astype('int64')
	newdata = replacing_missing_values(csvdata)
	newdata.to_csv("merged.csv",encoding='utf-8')
	return 0
	#else:
	#	print('not')

changing_datatypes(frame)


'''
all_data = pd.DataFrame()
for f in glob.glob(path + '/log*.csv'):
    df = pd.read_csv(f, parse_dates=[1])
    all_data = all_data.append(df, ignore_index=True)
'''
		#writer.writerows(records)



	















