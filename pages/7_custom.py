import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import matplotlib 
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)
##DURATION
##https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/TQCB/securities/RU000A1026B3.xml?from=2024-04-01&till=2024-04-11

mymap = ["#0051CA", "#F8AC27", "#3F863F", "#C6DBA1", "#FDD65F", "#FBEEBD", "#50766E"];

def sovdb_read(ticker, date):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \"""Date\""">='"+date.strftime('%Y-%m-%d')+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)
    df = pd.DataFrame(df).set_index('Date')
    df.index = pd.to_datetime(df.index)    
    df = df.sort_index()
    return df

def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

#get all countries and its keys
df_countr = sovdb_read_gen("countries")
m_keys = df_countr.m_key
countries = df_countr.name


INDICATORS = ['NGDP_RPCH','PCPIPCH','GGXWDG_NGDP','GGXCNL_NGDP','GGXONLB_NGDP','BCA_NGDPD']
INDICATORS_names = ['growth','cpi','gg debt, %gdp','balance, %GDP','pr balance, %GDP','ca, %GDP']


cols=st.columns(2)
with cols[0]:
    cntr = st.selectbox("Country:",countries, index=10)
with cols[1]:
    indic_name = st.selectbox("Indicator:",INDICATORS_names, index=0)

indic = INDICATORS[INDICATORS_names.index(indic_name)]
key = df_countr[df_countr.name==cntr].m_key.values[0]

Vintages = ["1021","0423","1023"]
Legend = ["Oct, 21","Apr, 23","Oct, 23","Apr, 24"]
title = cntr+", "+indic_name
    
tickers = []
for vintage in Vintages:
    tickers.append(key+"_"+indic+"_Y_WEO_"+vintage)
tickers.append(key+"_"+indic+"_Y_WEO")

date = pd.to_datetime('2010-12-31')
data0 = pd.DataFrame()
for tick in tickers:
    temp0 = sovdb_read(tick, date)
    temp = temp0['Value']
    temp = temp.rename(tick).to_frame()
    data0 = pd.concat([data0, temp],axis=1,)    
   
fig, ax = plt.subplots()
i=0
for ticker in tickers:
    if i==len(tickers)-1:
        ax.plot(data0[ticker],color=(1,0,0), linestyle = '--')
    else:
        ax.plot(data0[ticker],color=mymap[i])
    i=i+1

ax.legend(Legend)
ax.axvline(x=pd.to_datetime('2024-12-31'), color=(0.45, 0.45, 0.45), linestyle='--')
plt.title(title)
ax.set_xlabel("")
#ax.text(pd.to_datetime('2017-12-31'), -14, '@CBRSunnyMorning, Source: IMF WEO', fontsize=10,fontstyle='italic')
ax.axhline(0, color=(0.65,0.65,0.65))
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)

plt.show() 
st.pyplot(fig)

fn = key+"_"+indic+".png"
plt.savefig(fn)
with open(fn, "rb") as img:
    btn = st.download_button(
        label="JPG",
        data=img,
        file_name=fn,
        mime="image/png"
    )
    
