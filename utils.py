import pandas as pd
import io
import requests
import calendar as cal
monthsstr = list(cal.month_abbr)
date_today = pd.Timestamp.today().date()
filename = 'data_'+str(date_today.year)+'-'+monthsstr[date_today.month]+'-'+str(date_today.day)+'.csv'



url="https://coronavirus.data.gov.uk/api/v1/"+filename
s=requests.get(url).content
c=pd.read_csv(io.StringIO(s.decode('utf-8')))