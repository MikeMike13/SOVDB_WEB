import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import matplotlib 
import matplotlib.pyplot as plt
from datetime import datetime

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

def sovdb_read(ticker, date):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \"""Date\""">='"+date.strftime('%Y-%m-%d')+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)
    df = pd.DataFrame(df).set_index('Date')
    df.index = pd.to_datetime(df.index)    
    return df

def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

all_bonds = sovdb_read_gen('bonds')

cols=st.columns(3)
with cols[0]:
    ticker0 = st.selectbox("Choose bond: ",(all_bonds['rus_short']), index=0)
    temp = all_bonds[all_bonds['rus_short']==ticker0]
    ticker      = temp['id'].array[0]
    ticker_isin = temp['isin'].array[0]    
    
    
with cols[1]:
    date = st.date_input("Date: ", pd.to_datetime('2022-01-01'))  
with cols[2]:
    field0 = st.selectbox("plot",("Yield_Close","Price_Close"), index=0)        

#get data for selected bond
df = sovdb_read(ticker,date)
df = df[['Price_Close','Yield_Close','Volume']]
df = df.rename(columns={"Price_Close": ticker_isin+"_Price_Close", "Yield_Close": ticker_isin+"_Yield_Close", "Volume": ticker_isin+"_Vol"})
field = ticker_isin+"_"+field0

#get des of selected bond
query_des = "SELECT * FROM sovdb_schema.""bonds"" WHERE ""id""='"+ticker+"'"
#st.write(query_des)
cur = conn.cursor()
cur.execute(query_des);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df_des = pd.DataFrame(rows,columns=colnames)

cols=st.columns(4)
with cols[0]:
    name = st.write(df_des.rus_long.values[0])
with cols[1]:
    cpn = st.write("cpn: "+str(round(df_des.cpn_rate.values[0],2))+"%")
with cols[2]:
    mat = st.write("maturity: "+str(df_des.maturity_date.values[0]))
    #mat = datetime.strptime(df_des.maturity_date.values[0], '%y-%m-%d')
    #st.write(type(mat))
with cols[3]:    
    years = (df_des.maturity_date.values[0] - date.today()).days/365.25
    years_mat = st.write("years to mat: "+str(round(years,2))+"Y")

#plot selected bond    
fig, ax = plt.subplots()
Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
ax.plot(df[field], color=mymap[0], label='d',linewidth=0.8) 
ax.text(df[field].index[-1], df[field][-1], round(df[field][-1],2), fontsize=8,color=mymap[0]);#

plt.title(ticker+", "+Lastdate) 
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show() 
st.pyplot(fig)

ticker0_vs = st.selectbox("choose peer",(all_bonds['rus_short']), index=1)  
temp = all_bonds[all_bonds['rus_short']==ticker0_vs]
ticker_peer      = temp['id'].array[0]
ticker_isin_peer = temp['isin'].array[0]

df_peer = sovdb_read(ticker_peer,date)
df_peer = df_peer[['Price_Close','Yield_Close','Volume']]
df_peer = df_peer.rename(columns={"Price_Close": ticker_isin_peer+"_Price_Close", "Yield_Close": ticker_isin_peer+"_Yield_Close", "Volume": ticker_isin_peer+"_Vol"})
df_all = pd.concat([df, df_peer],axis=1, join="inner")  
df_all['Spread'] = (df_all[ticker_isin+"_Yield_Close"] - df_all[ticker_isin_peer+"_Yield_Close"])*100
field_peer = ticker_isin_peer+"_"+field0

cols=st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.plot(df_all[field], color=mymap[0], label=ticker0,linewidth=0.8) 
    ax.text(df_all[field].index[-1], df_all[field][-1], round(df_all[field][-1],2), fontsize=8,color=mymap[0]);#
    
    ax.plot(df_all[field_peer], color=mymap[1], label=ticker0_vs,linewidth=0.8) 
    ax.text(df_all[field_peer].index[-1], df_all[field_peer][-1], round(df_all[field][-1],2), fontsize=8,color=mymap[1]);#

    plt.title(ticker0+" vs "+ticker0_vs+", "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)
with cols[1]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.fill_between(df_all.index,df_all['Spread'], color=mymap[0], label='Spread',linewidth=0.8) 
    ax.text(df_all['Spread'].index[-1], df_all['Spread'][-1], round(df_all['Spread'][-1],0), fontsize=8,color=mymap[0]);#
        
    plt.title(ticker0+" vs "+ticker0_vs+", Spread,"+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)

cols=st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    ax.scatter(df[ticker_isin+"_Yield_Close"],df[ticker_isin+"_Price_Close"],color=mymap[0], s=10,alpha=0.5)
    ax.scatter(df[ticker_isin+"_Yield_Close"][-1],df[ticker_isin+"_Price_Close"][-1],color=(1,0,0), s=10)
    ax.set_xlabel('yield')
    ax.set_ylabel('price')
    plt.show() 
    st.pyplot(fig)    
fields_to_show = ['isin','rus_short','maturity_date']
st.write(all_bonds[fields_to_show])