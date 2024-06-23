import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from datetime import datetime
import io

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

def ticker_exists(ticker):
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker+"'"    
     cur = conn.cursor()
     cur.execute(query_s)
     rows = cur.fetchall()
     rows = np.array([*rows])
     return rows.size !=0
 
### IMF FC ###

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
    
### indicator by perrs ###
#rm NaN

df = sovdb_read_gen("peers")
peers = df.p_key

fig, ax = plt.subplots()
i=1
cols=st.columns(4)
with cols[0]:    
    indicator = st.selectbox("Indicator",("LCUSD","Key rate","CPI"), index=0)
with cols[1]:    
    date_st = st.date_input("From: ", pd.to_datetime('2022-12-29'))
with cols[2]:    
    peers = st.selectbox("Peers",peers, index=0)
with cols[3]:
    norm_100 = st.checkbox('norm',1) 

peers_sm_key = "PP_"+peers    
df = sovdb_read_gen(peers_sm_key)
peers_s_keys = df.m_key 

if indicator == "LCUSD":
    ticker_suff = "_LCUSD_D"
    title_name = "LCUSD, norm"
if indicator == "Key rate":
    ticker_suff = "_KEYRATE_D"
    title_name = "Key rates"
    

for key in peers_s_keys:
    ticker_x = key+ticker_suff
    is_x = ticker_exists(ticker_x)
    
    if is_x:             
        df_x = sovdb_read(ticker_x, date_st)                                    
        if df_x.empty:                            
            j=1
        else:
            df_x = df_x.rename(columns={"Value": key})                
            if i==1:
                df_f = df_x
                i=i+1
            else:
                df_f = pd.concat([df_f, df_x], axis=1, ignore_index=False, sort=True, )
                i=i+1
if norm_100:
    df_f = 100*(df_f / df_f.iloc[0, :])
    ax.axhline(100, color=(0.15,0.15,0.15))
for col in df_f.columns:
    df_temp = df_f[col].dropna() 
    line, = ax.plot(df_temp)    
    ax.annotate(col, (df_temp.index[-1], df_temp[-1]),color = line.get_color(),ha='left', va='bottom', size=8)                

 
ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
plt.title(title_name)    
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show()     
st.pyplot(fig)

cols = st.columns(6)
with cols[0]: 
    fn="PEERS-"+peers+"-indic-"+indicator+".png"
    plt.savefig(fn)
    with open(fn, "rb") as img:
        btn = st.download_button(
            label="JPG",
            data=img,
            file_name=fn,
            mime="image/png"
        )
        
with cols[1]:     
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
        df_f.to_excel(writer, sheet_name='Sheet1', index=True)    
    download2 = st.download_button(
        label="Excel",
        data=buffer,
        file_name="PEERS-"+peers+"-indic-"+indicator+".xlsx",
        mime='application/vnd.ms-excel'
    )
    
#COMMOD
date_st_c = st.date_input("Start: ", pd.to_datetime('2022-12-29'))
df_com = sovdb_read("CMDT_OILBRENT_CBONDS", date_st_c)     
field = "Value"
fig, ax = plt.subplots()    
line, = ax.plot(df_com)
ax.text(df_com[field].index[-1], df_com[field][-1], round(df_com[field][-1],2), fontsize=8,color=mymap[0]);#
plt.title("Brent, "+str(df_com[field].index[-1]))
ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--') 
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show()     
st.pyplot(fig)

cols = st.columns(6)
#with cols[0]: 
    #fn="PEERS-"+peers+"-indic-"+indicator+".png"
    #plt.savefig(fn)
    #with open(fn, "rb") as img:
    #    btn = st.download_button(
    #        label="JPG",
    #        data=img,
    #        file_name=fn,
    #        mime="image/png"
    #    )
        
with cols[0]:     
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
        df_com.to_excel(writer, sheet_name='Sheet1', index=True)    
    download2 = st.download_button(
        label="Excel",
        data=buffer,
        file_name="Brent.xlsx",
        mime='application/vnd.ms-excel'
    )