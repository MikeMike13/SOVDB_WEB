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
d1 = timedelta(days=2*30)

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

with cols[1]:
     ticker1 = "GDPN_Q_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
             
     if is_t1:
         #indicator2                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp                                    
         
         p1 =ax.bar(df_1.index, df_1[ticker1],width=d1, color=mymap[0],label='mom, rhs')
         #handles_t.append(p2)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#           
    
         plt.title("GDP, bln USD, "+df_1.index[-1].strftime("%B,%Y"))              
       
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
     ticker1 = "IP_M"
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

         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='industrial production')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         

         
         plt.title(countr+". Industry, SA, index, "+df_1.index[-1].strftime("%B,%Y"))                           
    
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()
     #ax.legend(handles=[p1, p2])  
     st.pyplot(fig)  
     
     
with cols[1]:
     ticker1 = "IP_M_YOY"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "IPMNF_M_YOY"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    
     
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)

     if is_t1:         
         #indicator1
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1}) 
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp         

         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='industrial production')          
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         

     if is_t2:         
         #ax2 = ax.twinx()
         #indicator2
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2}) 
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp         
         
         
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
                
        if is_t1:
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
         
         p1 =ax.bar(df_1.index, df_1[ticker1],width=d, color=mymap[0],label='revenues')
         #p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Revenues')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp           
         exp_12M = df_2.rolling(12).sum()
         
         #p2, = ax.plot(df_2, color=mymap[1],label='Expenditires')
         p2 =ax.bar(df_2.index+timedelta(days=12), df_2[ticker2],width=d, color=mymap[1],label='expenditures')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title(countr+". Government balance, bln LC, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
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
    
         plt.title(countr+". Government balance, 12M sum, bln LC, "+df_1.index[-1].strftime("%B,%Y"))              
       
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
    
         plt.title(countr+". Budget liquidity, bln LC, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 

         
st.subheader('External')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "CA_Q_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
     is_ca = is_t1
 
     ticker2 = "KA_Q_USD"
     ticker2_sel = key+"_"+ticker2
   
     ticker3 = "FA_Q_USD"
     ticker3_sel = key+"_"+ticker3

     ticker4 = "EO_Q_USD"
     ticker4_sel = key+"_"+ticker4

     ticker4a = "IMPGS_Q_USD"
     ticker4a_sel = key+"_"+ticker4a
          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp   
         CA = df_1
         
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp   
         KA = df_2
         
         temp = (-1)*sovdb_read(ticker3_sel, short_date)
         temp = temp.rename(columns={"Value": ticker3})    
         macro_data = macro_data.join(temp, how="outer")
         df_3 = temp   
         FA = df_3
         
         temp = sovdb_read(ticker4_sel, short_date)
         temp = temp.rename(columns={"Value": ticker4})    
         macro_data = macro_data.join(temp, how="outer")
         df_4 = temp   
         EO = df_4         
         
         temp = sovdb_read(ticker4a_sel, short_date)
         temp = temp.rename(columns={"Value": ticker4a})    
         #macro_data = macro_data.join(temp, how="outer")
         df_4a = temp   
         IMPGS = df_4a
         IMPGS_4Q = IMPGS.rolling(4).sum()
         
         
         BOP_tbl = CA.join(KA).join(FA).join(EO)                           
         #st.write(BOP_tbl)
         plot = BOP_tbl.plot(kind="bar", stacked=True, ax=ax,color=mymap[:4])         
         ax.legend(["CA","KA","FA","EO"]);                           
    
         plt.title(countr+". Balance of payments, bln USD, "+BOP_tbl.index[-1].strftime("%B,%Y"))         
         
         ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%y'))        
         
         #ax.set_xticklabels(labels=BOP_tbl.index.to_list().strftime('%b-%y'))
         #datetime.today().strftime('%Y-%m-%d')
       
     #formatter = matplotlib.dates.DateFormatter('%b-%y')
     #ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig) 
     
with cols[1]:
     ticker1 = "BG_Q_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "BS_Q_USD"
     ticker2_sel = key+"_"+ticker2
   
     ticker3 = "PI_Q_USD"
     ticker3_sel = key+"_"+ticker3

     ticker4 = "SI_Q_USD"
     ticker4_sel = key+"_"+ticker4

          
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)     
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp   
         BG = df_1
         
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp   
         BS = df_2
         
         temp = (-1)*sovdb_read(ticker3_sel, short_date)
         temp = temp.rename(columns={"Value": ticker3})    
         macro_data = macro_data.join(temp, how="outer")
         df_3 = temp   
         PI = df_3
         
         temp = sovdb_read(ticker4_sel, short_date)
         temp = temp.rename(columns={"Value": ticker4})    
         macro_data = macro_data.join(temp, how="outer")
         df_4 = temp   
         SI = df_4         
         
         CA_tbl = BG.join(BS).join(PI).join(SI)
         
         CA_tbl.plot(kind='bar', stacked=True, ax=ax,color=mymap[:4])         
         #CA.plot(ax=ax, color='r', linewidth=0.8)                           
         ax.set_xticklabels(labels = CA_tbl.index)
         
         #ax.plot(CA, color=mymap[0], linewidth=0.8)     
         #ax.set_xticklabels(labels = CA.index)
         
         ax.legend(["Goods","Services","Primary","Secondary"]);                               
         plt.title(countr+". Currnet account bln USD, "+CA_tbl.index[-1].strftime("%B,%Y"))                     
         
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     

     st.pyplot(fig) 

cols=st.columns(2)        
with cols[0]:
     ticker1 = "FDI_Q_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "PORT_Q_USD"
     ticker2_sel = key+"_"+ticker2
   
     ticker3 = "DER_Q_USD"
     ticker3_sel = key+"_"+ticker3

     ticker4 = "OTH_Q_USD"
     ticker4_sel = key+"_"+ticker4
     
     ticker5 = "CARES_Q_USD"
     ticker5_sel = key+"_"+ticker5
     
     ticker6 = "EXEPFIN_Q_USD"
     ticker6_sel = key+"_"+ticker6          

     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_t1:
         #indicator1                  
         temp = sovdb_read(ticker1_sel, short_date)
         temp = temp.rename(columns={"Value": ticker1})    
         macro_data = macro_data.join(temp, how="outer")
         df_1 = temp   
         FDI = df_1
         
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp   
         PORT = df_2
         
         temp = (-1)*sovdb_read(ticker3_sel, short_date)
         temp = temp.rename(columns={"Value": ticker3})    
         macro_data = macro_data.join(temp, how="outer")
         df_3 = temp   
         DER = df_3
         
         temp = sovdb_read(ticker4_sel, short_date)
         temp = temp.rename(columns={"Value": ticker4})    
         macro_data = macro_data.join(temp, how="outer")
         df_4 = temp   
         OTH = df_4    
         
         temp = sovdb_read(ticker5_sel, short_date)
         temp = temp.rename(columns={"Value": ticker5})    
         macro_data = macro_data.join(temp, how="outer")
         df_5 = temp   
         RES = df_5      
         
         temp = sovdb_read(ticker6_sel, short_date)
         temp = temp.rename(columns={"Value": ticker6})    
         macro_data = macro_data.join(temp, how="outer")
         df_6 = temp   
         EXFIN = df_6      
         
                  
         FA_tbl = FDI.join(PORT).join(DER).join(OTH).join(RES).join(EXFIN)                           
         #st.write(BOP_tbl)
         plot = FA_tbl.plot(kind="bar", stacked=True, ax=ax,color=mymap[:6])         
         ax.legend(["FDI","Portfolio","Derivatives","Other","Reserves","Exceptional, incl IMF"]);                           
    
         plt.title(countr+". Financial account, bln USD, "+FA_tbl.index[-1].strftime("%B,%Y"))         
         
         ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b-%y'))        
         
         #ax.set_xticklabels(labels=BOP_tbl.index.to_list().strftime('%b-%y'))
         #datetime.today().strftime('%Y-%m-%d')
       
     #formatter = matplotlib.dates.DateFormatter('%b-%y')
     #ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig) 

with cols[1]:
     fig = plt.figure()
     ax = fig.add_subplot(1, 1, 1)
     
     handles_t = [];
     if is_ca:

         p1, =ax.plot(CA, color=mymap[0], linewidth=0.8,label='CA')     
         handles_t.append(p1)
         #ax.text(CA.index[-1], exp_12M.values[-1][0], round(exp_12M.values[-1][0],2), fontsize=8,color=mymap[0])#
         
         p2, = ax.plot(BG, color=mymap[1],label='Goods')
         handles_t.append(p2)
         #ax.text(imp_12M.index[-1], imp_12M.values[-1][0], round(imp_12M.values[-1][0],2), fontsize=8,color=mymap[1])#           
         ax2 = ax.twinx()
         p3, = ax2.plot(1/lcusd_m_avg, color='r',label='LCUSD, inv, rhs')
         handles_t.append(p3)
         
         plt.title(countr+". FX vs external balances, "+CA.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig)

     
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
         
         #p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Exports')     
         p1 =ax.bar(df_1.index, df_1[ticker1],width=d, color=mymap[0],label='exports')
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp           
         imp_12M = df_2.rolling(12).sum()
         
         #p2, = ax.plot(df_2, color=mymap[1],label='Imports')
         p2 =ax.bar(df_2.index+timedelta(days=12), df_2[ticker2],width=d, color=mymap[1],label='imports')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title(countr+". Trade of goods, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
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
    
         plt.title(countr+". Trade of goods, 12M sum, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
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
         
     plt.title(countr+". Reserves, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
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
                      
         plt.title(countr+".Reserves, gold, tonns, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig) 

cols=st.columns(2)        
with cols[0]:
     ticker1 = "RES_Q_USD"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "EXTD_Q_USD"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)    

     ticker3 = "EXTD1Y_Q_USD"
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
         res_q = df_1         
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='Reserves')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})             
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp                    
         extd_q = df_2
         #st.write(extd_q)
         
         p2, = ax.plot(df_2, color=mymap[1],label='External debt')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
     if is_t3:
         #indicator3                  
         temp = sovdb_read(ticker3_sel, short_date)
         temp = temp.rename(columns={"Value": ticker3})    
         macro_data = macro_data.join(temp, how="outer")
         df_3 = temp        
         extd1y_q = df_3
         #st.write(extd1y_q)
         
         p3, = ax.plot(df_3, color=mymap[2],label='External Debt, 1Y due')
         handles_t.append(p3)
         ax.text(df_3.index[-1], df_3.values[-1][0], round(df_3.values[-1][0],2), fontsize=8,color=mymap[2])#      
         
         plt.title(countr+". Reserves & Debt, bln USD, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 


with cols[1]:
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
        
    
    handles_t = [];
    if is_t1 & is_t2:
        
        ext_cover = res_q
        
        ext_cover = ext_cover.join(extd_q, how="outer")
        ext_cover = ext_cover.join(extd1y_q, how="outer")
        ext_cover = ext_cover.join(IMPGS_4Q, how="outer")
        
        ext_cover['res_extd'] = ext_cover['RES_Q_USD']/ext_cover['EXTD_Q_USD']*100
        ext_cover['res_extdY'] = ext_cover['RES_Q_USD']/ext_cover['EXTD1Y_Q_USD']*100
        ext_cover['res_imp'] = ext_cover['RES_Q_USD']/(ext_cover['IMPGS_Q_USD']/12)
        
        #st.write(ext_cover['res_extdY'].dropna().values)
        p1, =ax.plot(ext_cover['res_extd'], color=mymap[0], linewidth=0.8,label='Reserves / External debt')   
        handles_t.append(p1)
        
        ax2 = ax.twinx()
        p2, = ax2.plot(ext_cover['res_imp'], color=mymap[1],label='Months of import G&S, rhs')
        handles_t.append(p2)
        
        plt.suptitle(countr+". Coverage ratios")   
        plt.title("1Y due covered by "+str(round(ext_cover['res_extdY'].dropna().values[-1]/100,1))+"x times")   
    
    formatter = matplotlib.dates.DateFormatter('%b-%y')
    ax.xaxis.set_major_formatter(formatter)
    plt.show()     
    ax.legend(handles=handles_t)  
    st.pyplot(fig)
     
st.subheader('Banks')
cols=st.columns(2)
with cols[0]:
     ticker1 = "ROE_Q"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "ROA_Q"
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
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='ROE')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         
         df_2 = temp                    
         ax2 = ax.twinx()
         p2, = ax2.plot(df_2, color=mymap[1],label='ROA')
         handles_t.append(p2)
         ax2.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title(countr+". Profitability "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 




st.subheader('Markets')
cols=st.columns(2)       
with cols[0]:
     ticker1 = "LCUSD_M_EOP"
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
         lsusd = df_1
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8)     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#         
         plt.suptitle(countr+". LCUSD, eop, "+df_1.index[-1].strftime("%B,%Y"))                  
             
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     #ax.legend(handles=handles_t)  
     st.pyplot(fig) 
     
with cols[1]:
     ticker1 = "NEER_M"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
 
     ticker2 = "REER_M"
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
         neer = df_1
         
         p1, =ax.plot(df_1, color=mymap[0], linewidth=0.8,label='NEER')     
         handles_t.append(p1)
         ax.text(df_1.index[-1], df_1.values[-1][0], round(df_1.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
     if is_t2:
         #indicator2                  
         temp = sovdb_read(ticker2_sel, short_date)
         temp = temp.rename(columns={"Value": ticker2})    
         macro_data = macro_data.join(temp, how="outer")
         df_2 = temp           
         reer = df_2
         
         p2, = ax.plot(df_2, color=mymap[1],label='REER')
         handles_t.append(p2)
         ax.text(df_2.index[-1], df_2.values[-1][0], round(df_2.values[-1][0],2), fontsize=8,color=mymap[1])#           
    
         plt.title(countr+". Effective FX rate, "+df_1.index[-1].strftime("%B,%Y"))         
     
       
     formatter = matplotlib.dates.DateFormatter('%b-%y')
     ax.xaxis.set_major_formatter(formatter)
     plt.show()     
     ax.legend(handles=handles_t)  
     st.pyplot(fig) 
     
cols=st.columns(2)            
with cols[0]:
     if is_t1:
         neer_norm = 100*(neer / neer.iloc[0, :])
         reer_norm = 100*(reer / reer.iloc[0, :])
         lcusd_inf = 1/lcusd_m_eop
         lcusd_norm = 100*(lcusd_inf / lcusd_inf.iloc[0, :])
              
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         handles_t = [];
         p1, =ax.plot(neer_norm, color=mymap[0], linewidth=0.8,label='NEER')     
         handles_t.append(p1)
         ax.text(neer_norm.index[-1], neer_norm.values[-1][0], round(neer_norm.values[-1][0],2), fontsize=8,color=mymap[0])#                  
             
         p2, = ax.plot(reer_norm, color=mymap[1],label='REER')
         handles_t.append(p2)
         ax.text(reer_norm.index[-1], reer_norm.values[-1][0], round(reer_norm.values[-1][0],2), fontsize=8,color=mymap[1])#           
        
         p3, = ax.plot(lcusd_norm, color=mymap[2],label='LCUSD')
         handles_t.append(p3)
         ax.text(lcusd_norm.index[-1], lcusd_norm.values[-1][0], round(lcusd_norm.values[-1][0],2), fontsize=8,color=mymap[2])#           
         ax.axhline(y=100, color = (0.5, 0.5, 0.5))
        
         plt.title(countr+". FX norm, "+reer_norm.index[-1].strftime("%B,%Y"))         
         
           
         formatter = matplotlib.dates.DateFormatter('%b-%y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show()     
         ax.legend(handles=handles_t)  
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
