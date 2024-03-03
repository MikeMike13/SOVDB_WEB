import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, datetime

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
with cols[2]:
    short_date = st.date_input("From date (markets): ", pd.to_datetime('2021-12-31'))
    
st.subheader('Markets')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "LCUSD_D"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])    
         
     if rows_1x.size !=0 :
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         query = "SELECT * FROM sovdb_schema.\""+ticker1_sel+"\" WHERE \"""Date\""">='"+short_date.strftime('%Y-%m-%d')+"'"   
         cur.execute(query);
         rows = cur.fetchall()
         colnames = [desc[0] for desc in cur.description]
         df_1_d = pd.DataFrame(rows,columns=colnames)
         df_1_d = pd.DataFrame(df_1_d).set_index('Date')
         df_1_d.index = pd.to_datetime(df_1_d.index)         

         df_y = df_1_d.resample('Y').last()
         YTD_FX = (df_y.values[-1][0]/df_y.values[-2][0]-1)*100
         #st.write(df_y.values[-2][0])
         #Lastdate = df.index[-1].strftime('%Y-%m-%d')
         #st.write(colnames)
         ax.plot(df_1_d, color=mymap[0], linewidth=0.8)          
         ax.text(df_1_d.index[-1], df_1_d.values[-1][0], round(df_1_d.values[-1][0],2), fontsize=8,color=mymap[0])#
         #ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.plot(df_y.index[-2], df_y.values[-2][0], marker=5,color=(1,0,0)) 
         #ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
              
             
         plt.title("Local currency to USD (YTD:"+str(round(YTD_FX,2))+"%)") 
         #plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)         
         
         
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
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
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
         ax.text(df_1_d.index[-1], df_1_d.values[-1][0], round(df_1_d.values[-1][0],2), fontsize=8,color=mymap[0])#
         ax.plot(df_2_d, color=mymap[1], label='cpi, avg',linewidth=0.8) 
         ax.text(df_2_d.index[-1], df_2_d.values[-1][0], round(df_2_d.values[-1][0],2), fontsize=8,color=mymap[1])#
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
         
         if limit_y:
             mult1 = 3
             mult2 = 9
             y_lim_up = min(df_1_d.median()[0]*mult2,df_2_d.median()[0]*mult1)
             y_lim_down = max(df_1_d.median()[0]*(-2),df_2_d.median()[0]*(-1))
             plt.ylim(y_lim_down, y_lim_up)
              
             
         plt.title("Growth vs inflation") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
with cols[1]:
     ticker1 = "NGDP_R_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "NGDPD_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])   
    
     
     if rows_1x.size !=0 and rows_2x.size != 0:
         #fig, ax = plt.subplots()
         #fig = plt.figure(figsize=(10,8))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         #plt.figure(figsize=(10,6))
         
                 
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
         p1 = ax.plot(df_1_d, color=mymap[0],  label='gdp, constant',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax2 = ax.twinx()
         p2 = ax2.plot(df_2_d, color=mymap[1], label='gdp, bln USD, rhs',linewidth=0.8)               
             
         plt.title("GDP: const vs USD") 
         p12 = p1+p2
         labs = [l.get_label() for l in p12]
         ax.legend(p12, labs, loc=4, frameon=False)
        # ax2.legend(['gdp, constant','gdp, bln USD, rhs'], loc=4) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)    

cols=st.columns(2)        
with cols[0]:
     ticker1 = "LP_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "DDFERTRATE_Y_CUST"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])      
    
     
     if rows_1x.size !=0 :
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
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
         
         p1 = ax.plot(df_1_d, color=mymap[0], label='population',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         pop_last = df_1_d.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         pop_10Y = df_1_d.loc[datetime(date.today().year-10, 12, 31).strftime('%Y-%m-%d')]
         pop_pch = ((pop_last.values[0]/pop_10Y.values[0])**(1/10)-1)*100
         ax.text(datetime(date.today().year-1, 12, 31), pop_last, "10Y: "+str(round(pop_pch,1))+"%", fontsize=8,color='r');
         ax2 = ax.twinx()
         p2 = ax2.plot(df_2_d, color=mymap[1], label='fertility rate, rhs',linewidth=0.8) 
             
         plt.title("Population, mln persons") 
         p12 = p1+p2
         labs = [l.get_label() for l in p12]
         ax.legend(p12, labs, loc=4, frameon=False)

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
with cols[1]:
     ticker1 = "PPPPC_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "NGDPDPC_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])      
    
     
     if rows_1x.size !=0 :
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
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
         
         p1 = ax.plot(df_1_d, color=mymap[0], label='PPP, internationl $',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         last1 = df_1_d.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         ax.text(datetime(date.today().year-1, 12, 31), last1, last1.values[0], fontsize=8,color=mymap[0]);
         ax2 = ax.twinx()
         p2 = ax2.plot(df_2_d, color=mymap[1], label='USD, rhs',linewidth=0.8) 
         last2 = df_2_d.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         ax2.text(datetime(date.today().year-1, 12, 31), last2, last2.values[0], fontsize=8,color=mymap[1]);
         
         plt.title("GDP per Capita") 
         p12 = p1+p2
         labs = [l.get_label() for l in p12]
         ax.legend(p12, labs, loc=4, frameon=False)

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
             
st.subheader('Fiscal')     
cols=st.columns(2)        
with cols[0]:
     ticker1 = "GGR_NGDP_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "GGX_NGDP_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])   
    
     
     if rows_1x.size !=0 and rows_2x.size != 0:
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
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
         ax.plot(df_1_d, color=mymap[0], label='revenues',linewidth=0.8) 
         ax.plot(df_2_d, color=mymap[1], label='expenditures',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         #ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
         
         #if limit_y:
         #    mult1 = 3
         #    mult2 = 9
         #    y_lim_up = min(df_1_d.median()[0]*mult2,df_2_d.median()[0]*mult1)
         #    y_lim_down = max(df_1_d.median()[0]*(-2),df_2_d.median()[0]*(-1))
         #    plt.ylim(y_lim_down, y_lim_up)
              
             
         plt.title("GG revenues vs expenditures, %GDP") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
with cols[1]:
     ticker1 = "GGXCNL_NGDP_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
     
     ticker2 = "GGXONLB_NGDP_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
     cur.execute(query_s)
     rows_2 = cur.fetchall()
     rows_2x = np.array([*rows_2])   
    
     
     if rows_1x.size !=0 and rows_2x.size != 0:
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
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
         ax.plot(df_1_d, color=mymap[0], label='balance',linewidth=0.8) 
         ax.plot(df_2_d, color=mymap[1], label='primary balance',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
         
         #if limit_y:
         #    mult1 = 3
         #    mult2 = 9
         #    y_lim_up = min(df_1_d.median()[0]*mult2,df_2_d.median()[0]*mult1)
         #    y_lim_down = max(df_1_d.median()[0]*(-2),df_2_d.median()[0]*(-1))
         #    plt.ylim(y_lim_down, y_lim_up)
              
             
         plt.title("GG balances, %GDP") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)

cols=st.columns(2)        
with cols[0]:
    ticker1 = "GGXWDG_NGDP_Y_WEO"
    #st.write(ticker1)
    ticker1_sel = key+"_"+ticker1
    #check if exists     
    query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
    cur.execute(query_s)
    rows_1 = cur.fetchall()
    rows_1x = np.array([*rows_1])
    
    #ticker2 = "NGDPDPC_Y_WEO"
    #ticker2_sel = key+"_"+ticker2
    #check if exists     
    #query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker2_sel+"'"    
    #cur.execute(query_s)
    #rows_2 = cur.fetchall()
    #rows_2x = np.array([*rows_2])      
    
    
    if rows_1x.size !=0 :
        #fig, ax = plt.subplots()
        #plt.figure(figsize=(10,6))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        query = "SELECT * FROM sovdb_schema.\""+ticker1_sel+"\""    
        cur.execute(query);
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        df_1_d = pd.DataFrame(rows,columns=colnames)
        df_1_d = pd.DataFrame(df_1_d).set_index('Date')
        df_1_d.index = pd.to_datetime(df_1_d.index)         
        df_1_d = df_1_d[(df_1_d.index >= st_date.strftime('%Y-%m-%d'))]
        #st.write(df_1_d)
        #query = "SELECT * FROM sovdb_schema.\""+ticker2_sel+"\""    
        #cur.execute(query);
        #rows = cur.fetchall()
        #colnames = [desc[0] for desc in cur.description]
        #df_2_d = pd.DataFrame(rows,columns=colnames)
        #df_2_d = pd.DataFrame(df_2_d).set_index('Date')
        #df_2_d.index = pd.to_datetime(df_2_d.index)
        #df_2_d = df_2_d[(df_2_d.index >= st_date.strftime('%Y-%m-%d'))]
        
        p1 = ax.plot(df_1_d, color=mymap[0], label='% GDP',linewidth=0.8) 
        ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
        last1 = df_1_d.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
        ax.text(datetime(date.today().year-1, 12, 31), last1, last1.values[0], fontsize=8,color=mymap[0]);
        #ax2 = ax.twinx()
        #p2 = ax2.plot(df_2_d, color=mymap[1], label='USD, rhs',linewidth=0.8) 
        #last2 = df_2_d.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
        #ax2.text(datetime(date.today().year-1, 12, 31), last2, last2.values[0], fontsize=8,color=mymap[1]);
        
        plt.title("GG Debt") 
        #p12 = p1+p2
        #labs = [l.get_label() for l in p12]
        #ax.legend(p12, labs, loc=4, frameon=False)
    
        formatter = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(formatter)
        plt.show() 
        st.pyplot(fig)
         
st.subheader('External')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "BCA_NGDPD_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     #check if exists     
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker1_sel+"'"    
     cur.execute(query_s)
     rows_1 = cur.fetchall()
     rows_1x = np.array([*rows_1])
      
    
     
     if rows_1x.size !=0 :
         #fig, ax = plt.subplots()
         #plt.figure(figsize=(10,6))
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         query = "SELECT * FROM sovdb_schema.\""+ticker1_sel+"\""    
         cur.execute(query);
         rows = cur.fetchall()
         colnames = [desc[0] for desc in cur.description]
         df_1_d = pd.DataFrame(rows,columns=colnames)
         df_1_d = pd.DataFrame(df_1_d).set_index('Date')
         df_1_d.index = pd.to_datetime(df_1_d.index)         
         df_1_d = df_1_d[(df_1_d.index >= st_date.strftime('%Y-%m-%d'))]       
     
         
         #Lastdate = df.index[-1].strftime('%Y-%m-%d')
         #st.write(colnames)
         ax.plot(df_1_d, color=mymap[0], label='revenues',linewidth=0.8)          
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
              
             
         plt.title("Current account balance, %GDP") 
         #plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)         