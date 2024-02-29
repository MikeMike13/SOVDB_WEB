import streamlit as st
import psycopg2 as ps
import pandas as pd
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

query = "SELECT * FROM sovdb_schema.countries"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
count_sel = df.name


df = df.fillna('') 

countr = st.selectbox("Country",(count_sel), index=203)
#st.write(df)
#st.write(df[df.name==countr].type.values[0])
key = df[df.name==countr].m_key.values[0]

st.header('Description data')
cols=st.columns(2)
with cols[0]:
    if (len(df[df.name==countr].type.values[0])==0):
        st.write("Type: none")
    else:
        st.write("Type: "+df[df.name==countr].type.values[0])
with cols[1]:
    if (len(df[df.name==countr].fx.values[0])==0):
        st.write("Currency: none")
    else:
        st.write("Currency: "+df[df.name==countr].fx.values[0])    
    
cols=st.columns(4)
with cols[0]:
    st.write("Master key (M_KEY): "+df[df.name==countr].m_key.values[0])
with cols[1]:
    if (len(df[df.name==countr].imf_key.values[0])==0):
        st.write("IMF key: none")
    else:
        st.write("IMF key: "+df[df.name==countr].imf_key.values[0])
with cols[2]:
    if (len(df[df.name==countr].wb_key.values[0])==0):
        st.write("WB key: none")
    else:
        st.write("WB key: "+df[df.name==countr].wb_key.values[0])      
with cols[3]:    
    if (len(df[df.name==countr].ecb_key.values[0])==0):
        st.write("ECB key: none")
    else:
        st.write("ECB key: "+df[df.name==countr].ecb_key.values[0])

cols=st.columns(4)
with cols[0]:
    if (len(df[df.name==countr].oecd_key.values[0])==0):
        st.write("OECD key: none")
    else:
        st.write("OECD key: "+df[df.name==countr].oecd_key.values[0])
with cols[1]:
    if (len(df[df.name==countr].bis_key.values[0])==0):
        st.write("BIS key: none")
    else:
        st.write("BIS key: "+df[df.name==countr].bis_key.values[0])
with cols[2]:
    if (len(df[df.name==countr].iso2_key.values[0])==0):
        st.write("ISO2 key: none")
    else:
        st.write("ISO2 key: "+df[df.name==countr].iso2_key.values[0])      
with cols[3]:    
    if (len(df[df.name==countr].iso3_key.values[0])==0):
        st.write("ISO3 key: none")
    else:
        st.write("ISO3 key: "+df[df.name==countr].iso3_key.values[0])

st.header('Map')
        
st.header('Key charts')
cols=st.columns(4)        
with cols[0]:
    limit_y = st.checkbox('auto limit y',1) 
with cols[1]:
    st_date = st.date_input("From date: ", pd.to_datetime('2000-12-31'))

st.subheader('Markets')

st.subheader('Macro')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "NGDP_RPCH_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "PCPIPCH_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])   
    
     
     if rows_1x.size !=0 and rows_2x.size != 0:
         fig, ax = plt.subplots()
         
         query = "SELECT * FROM sovdb_schema.\""+ticker1_sel+"\""    
         cur.execute(query);
         rows = cur.fetchall()
         colnames = [desc[0] for desc in cur.description]
         df_1_d = pd.DataFrame(rows,columns=colnames)
         df_1_d = pd.DataFrame(df_1_d).set_index('Date')
         df_1_d.index = pd.to_datetime(df_1_d.index)         
         df_1_d = df_1_d[(df_1_d.index >= st_date.strftime('%Y-%m-%d'))]
         
         query = "SELECT * FROM sovdb_schema.\""+ticker2_sel+"\""    
         cur.execute(query);
         rows = cur.fetchall()
         colnames = [desc[0] for desc in cur.description]
         df_2_d = pd.DataFrame(rows,columns=colnames)
         df_2_d = pd.DataFrame(df_2_d).set_index('Date')
         df_2_d.index = pd.to_datetime(df_2_d.index)
         df_2_d = df_2_d[(df_2_d.index >= st_date.strftime('%Y-%m-%d'))]
         
         #Lastdate = df.index[-1].strftime('%Y-%m-%d')
         #st.write(colnames)
         ax.plot(df_1_d, color=mymap[0], label='gdp growth',linewidth=0.8) 
         ax.plot(df_2_d, color=mymap[1], label='cpi, avg',linewidth=0.8) 
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
         
         if limit_y:
             mult1 = 3
             mult2 = 9
             y_lim_up = min(df_1_d.median()[0]*mult2,df_2_d.median()[0]*mult1)
             y_lim_down = max(df_1_d.median()[0]*(-2),df_2_d.median()[0]*(-1))
             plt.ylim(y_lim_down, y_lim_up)
              
             
         plt.title("Growth vs inflation") 
         plt.legend() 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
    
             
st.subheader('Fiscal')            
st.subheader('External')
         