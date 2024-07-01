import psycopg2 as ps
import pandas as pd
#import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, datetime
import io
import streamlit as st
from datetime import timedelta


st.set_page_config(layout="centered")

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

def sovdb_read(ticker, date):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \"""Date\""">='"+date.strftime('%Y-%m-%d')+"' ORDER by \"""Date\""""    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)
    df = pd.DataFrame(df).set_index('Date')
    df.index = pd.to_datetime(df.index)    
    return df

def sovdb_read_date(ticker, date):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \"""Date\"""='"+date.strftime('%Y-%m-%d')+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    rows = np.array([*rows])   
    if rows.size ==0:
        return 0
    else:
        return rows[0][1]

def sovdb_read_item(ticker, field, value):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \""+field+"\"='"+str(value)+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    rows = np.array([*rows])   
    if rows.size ==0:
        return 0
    else:
        return rows[0]

def ticker_exists(ticker):
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker+"'"    
     cur = conn.cursor()
     cur.execute(query_s)
     rows = cur.fetchall()
     rows = np.array([*rows])
     return rows.size !=0
 
def table_exists(ticker):
     #query_s = "SELECT * FROM sovdb_schema.\""+ticker+"\""
     query_s = "SELECT EXISTS (SELECT FROM pg_tables WHERE  schemaname = 'sovdb_schema' AND    tablename  = '"+ticker+"');"
     cur = conn.cursor()
     cur.execute(query_s)     
     rows = cur.fetchall()     
     return rows[0][0]
 
def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

#read all countries des
df_all = sovdb_read_gen("countries")
count_sel = df_all.name
df_all = df_all.fillna('') 

cols=st.columns(2)        
with cols[0]:
    countr = st.selectbox("Country",(count_sel), index=134)
    key = df_all[df_all.name==countr].m_key.values[0]
with cols[1]:
    short_date = st.date_input("From date: ", pd.to_datetime('2021-12-31'))

d = timedelta(days=10)

st.subheader('Real')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "CPI_M_YOY"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "CPI_M_MOM"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel) 
     
     
     if is_t1  & is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         data_1 = sovdb_read(ticker1_sel, short_date)
         data_1 = data_1.rename(columns={"Value": ticker1})         
         df_1 = data_1[ticker1].to_frame()
         
         ax2 = ax.twinx()
         #indicator2
         data_2 = sovdb_read(ticker2_sel, short_date)
         data_2 = data_2.rename(columns={"Value": ticker2})         
         df_2 = data_2[ticker2].to_frame()

         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='yoy')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         
         p2 =ax2.bar(df_2.index, df_2[ticker2],width=d, color=mymap[1],label='mom, rhs')
         ax2.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#  
         ax2.axvline(x=df_2.index[-12]-timedelta(days=15), color = mymap[1],linestyle='--')
         
         plt.title("CPI, "+df_1.index[-1].strftime("%B,%Y"))         
    
         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         ax.legend(handles=[p1, p2])  
         st.pyplot(fig)         

with cols[1]:
     ticker1 = "CPI_M_INDEX"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
     
     
     if is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         data_1 = sovdb_read(ticker1_sel, short_date)
         data_1 = data_1.rename(columns={"Value": ticker1})         
         df_1 = data_1[ticker1].to_frame()
         
         #df_1_y = df_1.resample('Y').last()
         #st.write(df_1.values[-12][0])
         ax.plot(df_1.index[-13], df_1.values[-13][0], marker=5,color=(1,0,0)) 
         
       
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='yoy')          

         plt.title("CPI, index "+df_1.index[-1].strftime("%B,%Y"))         
    
         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         st.pyplot(fig)         
 
