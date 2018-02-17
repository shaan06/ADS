import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys
import urllib.request
import csv
import os
import logging
import zipfile
import boto
import sys
import datetime
import time
from boto.s3.key import Key 
#from settings import PROJECT_ROOT


#All user Inputs
#Name = input("Name : ")
CIK = str(sys.argv[1]) 
acc_no = str(sys.argv[2]) 
AWS_ACCESS_KEY_ID = str(sys.argv[3])
AWS_SECRET_ACCESS_KEY = str(sys.argv[4])
#inputLocation = input("Bucket Location = ")

## Creating the Loggers
logger = logging.getLogger("Problem1")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("Prog1_Log.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


## Generating the URL

baseURL = "https://www.sec.gov"
if(str(CIK) != "" and str(acc_no) != ""):
	CIK = CIK.lstrip('0')
	def insert_dash(string):
	    if (len(string) == 18):
	        return string[:10] + "-" + string[10:12] + "-" + string[12:]

else:
	logger.warning("Invalid CIK or Accession_no")
	exit()

    
	
accession_no = insert_dash(acc_no)

def create_url(baseURL,CIK,acc_no,accession_no):
	return str(baseURL + "/Archives/edgar/data/"+ CIK+"/" + acc_no + "/"+ accession_no + "-index.html")
logger.info("Valid CIK and Accession_no provided")

logger.info("-index.html URL created")

URL = create_url(baseURL,CIK,acc_no,accession_no)
print(URL)
externalList = []
tablelist = []

def create_10q_url():
 	try:
		soup = BeautifulSoup((urllib.request.urlopen(URL)),"html.parser")
		for t in soup.find_all('table' , attrs={"summary": "Document Format Files"}):
		     for tr in soup.find_all('tr'):
		        for td in tr.findChildren('td'):
		            if(td.text == '10-Q'):
		                for a in tr.findChildren('a', href=True):
		                    return str(a['href'])


	logger.info("URL for the 10-q file for CIK = %s and Accession_no = %s is created", CIK,acc_no)
    except:
    	logger.warning("URL for the 10-q file for CIK = %s and Accession_no = %s is not found", CIK,acc_no)

                  
#print(finalURL)


finalURL = baseURL + create_10q_url() 
print(finalURL)
tablelist = []

soup = BeautifulSoup((urllib.request.urlopen(finalURL)),"html.parser")
tables = soup.find_all('table')
for table in tables: 
	for tr in table.find_all('tr'):
		i = 0
		for td in tr.findChildren('td'):
			if ("background" in str(td.get('style'))):
				tablelist.append(table)
				i = 1
				break
		if(i == 1):
			break	
logger.info("Relevant tables were appended")
#	print(len(tablelist))


#Creating directory for the csc files
x = '%s_all_csv' %CIK
if not os.path.exists(x):
    os.makedirs(x)



# Converting into CSV
for t in tablelist:
	records = []
	for tr in t.find_all('tr'):
		rString = []
		for td in tr.findAll('td'):
			p = td.findAll('p')
			if (len(p) > 0):
				for ps in p:
					ps_text = ps.get_text().replace("\n"," ") 
					ps_text = ps_text.replace("\xa0","")                 
					rString.append(ps_text)
			else:
				td_text=td.get_text().replace("\n"," ")
				td_text = td_text.replace("\xa0","")
				rString.append(td_text)
	
		records.append(rString)
	i = i + 1
	with open(os.path.join(x, str(i) + 'tables.csv'), 'wt') as f:
		writer = csv.writer(f)
		writer.writerows(records)

logger.info("CSV files for the tables were generated")



#Creating zip file for the CSVs and Log file
def zipping(path, ziph, tablelist):
    # ziph is zipfile handle
    j = 0
    for tab in tablelist:
    	j = j + 1
    	ziph.write(os.path.join(x, str(j)+'tables.csv'))
    ziph.write(os.path.join('Prog1_Log.log'))   


zf = zipfile.ZipFile('%s.zip' %CIK, 'w')
zipping('/', zf, tablelist)
zf.close()
logger.info("CSV files and the Log file were zipped in a single folder")

#Creating a S3 bucket
conn = boto.connect_s3(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

#Location for the region
'''loc = ''
if inputLocation == 'APNortheast':
    loc=boto.s3.connection.Location.APNortheast
elif inputLocation == 'APSoutheast':
    loc=boto.s3.connection.Location.APSoutheast
elif inputLocation == 'APSoutheast2':
    loc=boto.s3.connection.Location.APSoutheast2
elif inputLocation == 'CNNorth1':
    loc=boto.s3.connection.Location.CNNorth1
elif inputLocation == 'EUCentral1':
    loc=boto.s3.connection.Location.EUCentral1
elif inputLocation == 'EUWest2':
    loc=boto.s3.connection.Location.EU
elif inputLocation == 'SAEast':
    loc=boto.s3.connection.Location.SAEast
elif inputLocation == 'USWest':
    loc=boto.s3.connection.Location.USWest
elif inputLocation == 'USWest2':
    loc=boto.s3.connection.Location.USWest2'''

bucket_name = ts = time.time()
st = datetime.datetime.fromtimestamp(ts)    
bucket_name =  "statistical data " + str(CIK) + str(st).replace(" ", "").replace("-", "").replace(":","").replace(".","")
bucket = conn.create_bucket(bucket_name, location=boto.s3.connection.Location.DEFAULT)
filename = ('%s.zip' %CIK)
logger.info("S3 bucket successfully created")


#Uploading files to the Bucket
def percent_cb(complete, total):
	sys.stdout.write('.')
	sys.stdout.flush()

k = Key(bucket)
k.key = 'Problem1'
k.set_contents_from_filename(filename, cb=percent_cb, num_cb=10)

'''except:
    logger.warning("Invalid Amazon Keys.")
    print("Invalid Amazon Keys.")
    exit()'''
logger.info("Zip file successfully uploaded to S3")
print("Zip File successfully uploaded to S3")

logger.info("-----FINISH------")