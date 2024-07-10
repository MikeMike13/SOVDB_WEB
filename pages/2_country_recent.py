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


ticker1 = "LCUSD_M_AVG"
ticker1_sel = key+"_"+ticker1
temp= sovdb_read(ticker1_sel, short_date)
lcusd_m_avg = temp.rename(columns={"Value": ticker1})         

ticker2 = "LCUSD_M_EOP"
ticker2_sel = key+"_"+ticker2
temp = sovdb_read(ticker2_sel, short_date)
lcusd_m_eop = temp.rename(columns={"Value": ticker2})         

st.subheader('Real')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "GDPR_Q_YOY"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
     #indicator1
         macro_data = sovdb_read(ticker1_sel, short_date)
         macro_data = macro_data.rename(columns={"Value": ticker1})         
         df_1 = macro_data[ticker1].to_frame()         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='yoy')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         plt.suptitle(countr+". GDP growth, "+df_1.index[-1].strftime("%B,%Y"))                               
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig)    

cols=st.columns(2)       
with cols[0]:
     ticker1 = "CPI_M_YOY"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "CPI_M_MOM"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
     #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp
     
         #macro_data = sovdb_read(ticker1_sel, short_date)
         #macro_data = macro_data.rename(columns={"Value": ticker1})         
         #df_1 = macro_data[ticker1].to_frame()
         cpi_yoy = df_1
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='yoy')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         plt.suptitle(countr+". CPI, "+df_1.index[-1].strftime("%B,%Y"))                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp
           
         MAVG = np.mean(df_2.values[-3:])
         MAVG_ANN = ((1+MAVG/100)**12-1)*100    
                 
         ax2 = ax.twinx()
         p2 =ax2.bar(df_2.index, df_2[ticker2],width=d, color=mymap[1],label='mom, rhs')
         handles_t.append(p2)
         ax2.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#  
         ax2.axvline(x=df_2.index[-12]-timedelta(days=15), color = mymap[1],linestyle='--')
    
         plt.title("3M MA: "+str(round(MAVG,2))+"%, annualaized: "+str(round(MAVG_ANN,1))+"%")         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig)         

with cols[1]:
     ticker1 = "CPI_M_INDEX"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
     
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     if is_t1:
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})  
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp
         
         ax.plot(df_1.index[-13], df_1.values[-13][0], marker=5,color=(1,0,0))              
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='yoy')          
     
         plt.title(countr+". CPI, index "+df_1.index[-1].strftime("%B,%Y"))         
   
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()
     st.pyplot(fig)         
 
cols=st.columns(2)        
with cols[0]:
     ticker1 = "EMPL_M_PERS"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "UNEMPL_M_PERS"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
     
     if is_t1 & is_t2:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp         
         
         ax2 = ax.twinx()
         #indicator2
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2}) 
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='employed')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         
         p2, =ax2.plot(df_2, color=mymap[1], linewidth=0.8,label='unemployed, rhs')          
         ax2.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],1), fontsize=8,color=mymap[1])#  
         plt.title(countr+". Employment, mln persons, "+df_1.index[-1].strftime("%B,%Y"))                  
    
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         ax.legend(handles=[p1, p2])  
         st.pyplot(fig)    
         #st.

with cols[1]:
     ticker1 = "UNEMPL_M"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)  
          
     if is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)         
         
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})
         macro_data = macro_data.join(temp, how="outer")         
         df_1 = temp     
                
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8)          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],1), fontsize=8,color=mymap[0])#         
         
         plt.title(countr+". Unemployment rate, "+df_1.index[-1].strftime("%B,%Y"))                  
    
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()         
         st.pyplot(fig)  

cols=st.columns(2)        
with cols[0]:
     ticker1 = "IP_M_YOY"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "IPMNF_M_YOY"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
     
     if is_t1 & is_t2:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp         
         
         #ax2 = ax.twinx()
         #indicator2
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2}) 
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='industrial production')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         
         p2, =ax.plot(df_2, color=mymap[1], linewidth=0.8,label='manufacturing, rhs')          
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],1), fontsize=8,color=mymap[1])#  
         plt.title(countr+". Industry, yoy, "+df_1.index[-1].strftime("%B,%Y"))                  
         ax.axhline(y=0, color = (0.5, 0.5, 0.5))
         
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         ax.legend(handles=[p1, p2])  
         st.pyplot(fig)  
         
cols=st.columns(2)        
with cols[0]:
     ticker1 = "WAGES_M_LC"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
      
     
     #№ticker2 = "IPMNF_M_YOY"
     #ticker2_sel = key+"_"+ticker2
     #is_t2 = ticker_exists(ticker2_sel)    
     
     if is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp         
         wages_usd = df_1.values / lcusd_m_avg.values*1000
         wages_usd = pd.DataFrame(data=wages_usd, index=df_1.index)
                  
         df_2 = df_1.pct_change(periods=12) * 100  
         wages_yoy = df_2
         
         ax2 = ax.twinx()
         #indicator2
         #temp = sovdb_read(ticker2_sel, short_date)
         #temp = temp.rename(columns={"Value": ticker2}) 
         #macro_data = macro_data.join(temp, how="outer")
         #df_2 = temp         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='000, LC')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         
         p2, =ax2.plot(wages_usd, color=mymap[1], linewidth=0.8,label='USD')          
         ax2.text(wages_usd.index[-1], wages_usd.values[-1][0], round(wages_usd.values[-1][0],1), fontsize=8,color=mymap[1])#  
         plt.title(countr+". Wages, "+df_1.index[-1].strftime("%B,%Y"))                  
         #ax.axhline(y=0, color = (0.5, 0.5, 0.5))
         
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         ax.legend(handles=[p1, p2])  
         st.pyplot(fig)  

with cols[1]:    
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)                
                
        p1, =ax.plot(wages_yoy, color=mymap[0], linewidth=0.8,label='wages, yoy')          
        ax.text(wages_yoy.index[-1], wages_yoy.values[-1][0], round(wages_yoy.values[-1][0],2), fontsize=8,color=mymap[0])#         
        
        p2, =ax.plot(cpi_yoy, color=mymap[1], linewidth=0.8,label='cpi, yoy')          
        ax.text(cpi_yoy.index[-1], cpi_yoy.values[-1][0], round(cpi_yoy.values[-1][0],1), fontsize=8,color=mymap[1])#  
        plt.title(countr+". Wages, "+df_1.index[-1].strftime("%B,%Y"))                  
        #ax.axhline(y=0, color = (0.5, 0.5, 0.5))
        
        formatter = matplotlib.dates.DateFormatter('%b-%y')
        ax.xaxis.set_major_formatter(formatter)
        plt.show()
        ax.legend(handles=[p1, p2])  
        st.pyplot(fig) 
         
st.subheader('Fiscal')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "GOVREV_M_LC"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "GOVEXP_M_LC"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp   
         rev_12M = df_1.rolling(12).sum()         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Revenues')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp           
         exp_12M = df_2.rolling(12).sum()
         
         p2, = ax.plot(df_2, color=mymap[1],label='Expenditires')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title("Government balance, bln LC, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 
     
with cols[1]:
         
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:

         p1, =ax.plot(rev_12M, color=mymap[0], linewidth=0.8,label='Revenues')     
         handles_t.append(p1)
         ax.text(rev_12M.index[-1], rev_12M.values[-1][0], round(rev_12M.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:         
         
         p2, = ax.plot(exp_12M, color=mymap[1],label='Expenditires')
         handles_t.append(p2)
         ax.text(exp_12M.index[-1], exp_12M.values[-1][0], round(exp_12M.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title("Government balance, 12M sum, bln LC, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 
        
cols=st.columns(2)        
with cols[0]:
     ticker1 = "GOVDEBT_M_LC"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)  
     
     if is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp  
         
         df_1['govdebt_m_usd'] = df_1['GOVDEBT_M_LC'] / lcusd_m_eop['LCUSD_M_EOP']        
                  
         p1, =ax.plot(df_1['GOVDEBT_M_LC'], color=mymap[0], linewidth=0.8,label='bln LC')          
         ax2 = ax.twinx()         
         p2, =ax2.plot(df_1['govdebt_m_usd'], color=mymap[1], linewidth=0.8,label='bln USD, rhs')          
         ax2.text(df_1['govdebt_m_usd'].index[-1], df_1['govdebt_m_usd'].values[-1], round(df_1['govdebt_m_usd'].values[-1],1), fontsize=8,color=mymap[1])#  
                  
         plt.title(countr+". Government debt, bln LC, "+df_1.index[-1].strftime("%B,%Y"))                  
         
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()
         ax.legend(handles=[p1, p2])  
         st.pyplot(fig)  
         
with cols[1]:
     ticker1 = "CBLIABCG_M_LC"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "BNLIABCG_M_LC"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp            
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='in CBR')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp                    
         
         p2, = ax.plot(df_2, color=mymap[1],label='in Banks')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title("Budget liquidity, bln LC, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 

         
st.subheader('External')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "EXPG_M_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "IMPG_M_USD"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp   
         exp_12M = df_1.rolling(12).sum()         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Exports')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp           
         imp_12M = df_2.rolling(12).sum()
         
         p2, = ax.plot(df_2, color=mymap[1],label='Imports')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title("Trade of goods, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 
     
with cols[1]:
         
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:

         p1, =ax.plot(exp_12M, color=mymap[0], linewidth=0.8,label='Exports')     
         handles_t.append(p1)
         ax.text(exp_12M.index[-1], exp_12M.values[-1][0], round(exp_12M.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:         
         
         p2, = ax.plot(imp_12M, color=mymap[1],label='Imports')
         handles_t.append(p2)
         ax.text(imp_12M.index[-1], imp_12M.values[-1][0], round(imp_12M.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title("Trade of goods, 12M sum, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 

cols=st.columns(2)        
with cols[0]:
     ticker1 = "RES_M_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "RESFX_M_USD"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    

     ticker3 = "RESGOLD_M_USD"
     ticker3_sel = key+"_"+ticker3
     is_t3 = ticker_exists(ticker3_sel)  
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp            
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Total')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})             
         #macro_data = macro_data.join(temp, how="outer")
         df_2 = temp                    
         
         p2, = ax.plot(df_2, color=mymap[1],label='FX')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
     if is_t3:
         #indicator3                  
         temp = sovdb_read(ticker3_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_3 = temp        
         gold_res = df_3
         
         p3, = ax.plot(df_3, color=mymap[2],label='Gold')
         handles_t.append(p3)
         ax.text(df_3.index[-1], df_3.values[-1][0], round(df_3.values[-1][0],2), fontsize=8,color=mymap[2])#      
         
         plt.title("Reserves, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 

with cols[1]:
     ticker1 = "RESGOLD_M_OUNCE"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
               
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         temp = (temp*1000000)/35270 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Total')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
                      
         plt.title("Reserves, gold, tonns, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig) 

         
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
    macro_data.to_excel(writer, sheet_name='Sheet1', index=True)    
download = st.download_button(
    label="Excel",
    data=buffer,
    file_name=countr+"_recent_macro.xlsx",
    mime='application/vnd.ms-excel'
)         