import pandas as pd
import requests
from bs4 import BeautifulSoup
import sys
import urllib.request
import csv
import os
import logging
import zipfile
#from settings import PROJECT_ROOT

logger = logging.getLogger("Problem1")
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("Prog1_Log.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


CIK = input("CIK ") 
acc_no = input("Accession_no ")
baseURL = "https://www.sec.gov"
CIK = CIK.lstrip('0')
def insert_dash(string):
    if (len(string) == 18):
        return string[:10] + "-" + string[10:12] + "-" + string[12:]
    else:
        print("invailid")
        logger.warning("Invalid CIK or Accession_no")

logger.info("Valid CIK and Accession_no provided")
accession_no = insert_dash(acc_no)

def create_url(baseURL,CIK,acc_no,accession_no):
	return str(baseURL + "/Archives/edgar/data/"+ CIK+"/" + acc_no + "/"+ accession_no + "-index.html")

logger.info("-index.html URL created")

URL = create_url(baseURL,CIK,acc_no,accession_no)
print(URL)
externalList = []
tablelist = []

def create_10q_url():
	soup = BeautifulSoup((urllib.request.urlopen(URL)),"html.parser")
	for t in soup.find_all('table' , attrs={"summary": "Document Format Files"}):
	     for tr in soup.find_all('tr'):
	        for td in tr.findChildren('td'):
	            if(td.text == '10-Q'):
	                for a in tr.findChildren('a', href=True):
	                    return str(a['href'])

logger.info("URL for the 10-q file for CIK = %s and Accession_no = %s is created", CIK,acc_no)
            
                  
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
x = '%s_all_csv' %CIK
if not os.path.exists(x):
    os.makedirs(x)

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

def zipping(path, ziph, tablelist):
    # ziph is zipfile handle
    j = 0
    for tab in tablelist:
    	j = j + 1
    	ziph.write(os.path.join(x, str(j)+'tables.csv'))
    ziph.write(os.path.join('Prog1_Log.log'))   
    logger.info("CSV files and the Log file were zipped in a single folder")

zf = zipfile.ZipFile('%s.zip' %CIK, 'w')
zipping('/', zf, tablelist)
zf.close()



'''
rows = tablelist[0].find_all("tr")
print("=====================================================================================================================")
print (rows)
csvfile = open("yo2.csv","wt", newline ='')
writer = csv.writer(csvfile)
try:
	for row in rows:
		csvRow = []
		for cell in row.find_all('td'):
			print (cell.get_text())
			csvRow.append(cell.get_text())
			print(csvRow)
		writer.writerow(csvRow)
finally:
	csvfile.close()
'''
#line = (td.text.encode("utf-8")).decode('ascii', 'ignore')

  #  r = i['name']
  #  kuchbhi = finalURL + "#" + r
# soup = BeautifulSoup((urllib.request.urlopen(kuchbhi)), "html.parser")
#    for i, df in enumerate(pd.read_html(kuchbhi)):
#        df.to_csv('table%s.csv' % i)
       
    
    

  


    

                

