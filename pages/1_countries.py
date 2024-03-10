import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, datetime
import io

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

def sovdb_read_date(ticker, date):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \"""Date\"""='"+date.strftime('%Y-%m-%d')+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    rows = np.array([*rows])   
    return rows[0][1]

def ticker_exists(ticker):
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker+"'"    
     cur = conn.cursor()
     cur.execute(query_s)
     rows = cur.fetchall()
     rows = np.array([*rows])
     return rows.size !=0

def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df
     
df = sovdb_read_gen("countries")
count_sel = df.name

df = df.fillna('') 

countr = st.selectbox("Country",(count_sel), index=203)

key = df[df.name==countr].m_key.values[0]

#get all peers
df_p = sovdb_read_gen("peers")
peers = df_p.p_key

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

st.header('Rankings')
cols=st.columns(3) 
with cols[0]:
    date_rank = st.date_input("as of: ", pd.to_datetime('2022-12-31'))
with cols[1]:
    small_peers = st.selectbox("Peers",peers, index=0)
with cols[2]:
    calc_rank = st.checkbox('Calc rankings',0)    
    
if calc_rank:
    ticker_r1 = 'LP_Y_WEO'
    ticker_r10 = key+"_"+ticker_r1
    ticker_c_val = sovdb_read_date(ticker_r10, date_rank)
    #st.write(ticker_c_val)    
    
    peers_key = "PP_WEO"
    df = sovdb_read_gen(peers_key)        
    all_peers_keys = df.m_key 
    
    peers_key_sm = "PP_"+small_peers
    df = sovdb_read_gen(peers_key_sm)        
    small_peers_key = df.m_key 
    
    data_x = []    
    #get broad peers
    for key in all_peers_keys:
        ticker_x = key+"_"+ticker_r1
        is_x = ticker_exists(ticker_x)
                      
        if is_x:                                        
            df_x = sovdb_read_date(ticker_x, date_rank) 
            data_x.append(df_x)                   
            
    data_x = np.array(data_x)                
    data_x = np.sort(data_x)[::-1] 
                  
    rank1_all = np.where(data_x == ticker_c_val)
    rank1_all = rank1_all[0][0]+1
    
    data_x = [] 
    for key in small_peers_key:
        ticker_x = key+"_"+ticker_r1
        is_x = ticker_exists(ticker_x)
                      
        if is_x:                                        
            df_x = sovdb_read_date(ticker_x, date_rank) 
            data_x.append(df_x)                   
            
    data_x = np.array(data_x)                
    data_x = np.sort(data_x)[::-1] 
                  
    rank1_small = np.where(data_x == ticker_c_val)
    rank1_small = rank1_small[0][0]+1
    
    st.write("Population: #"+str(rank1_all)+" (peers #"+str(rank1_small)+")")
              

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
     is_t1 = ticker_exists(ticker1_sel) 
     
     if is_t1 :
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         macro_data = sovdb_read(ticker1_sel, short_date)
         macro_data = macro_data.rename(columns={"Value": ticker1})         
         df = macro_data[ticker1].to_frame()
         
         df_y = df.resample('Y').last()
         YTD_FX = (df_y.values[-1][0]/df_y.values[-2][0]-1)*100
         
         ax.plot(df, color=mymap[0], linewidth=0.8)          
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#         
         ax.plot(df_y.index[-2], df_y.values[-2][0], marker=5,color=(1,0,0)) 
                       
         End_val = df.values[-1][0]
         Start_val = df_y.values[-2][0]
         End_date_c = df.index[-1]
         Start_date_c = df_y.index[-2]

         period_ret = (End_val/Start_val-1)*100
         annula_ret = ((1+period_ret/100)**(365.25/(End_date_c - Start_date_c).days)-1)*100
         years = (End_date_c - Start_date_c).days/365.25       
                         
         plt.title("Local currency to USD "+str(df.index[-1].strftime('%Y-%m-%d'))+", YTD:"+str(round(YTD_FX,2))+"% (ann:"+str(round(annula_ret,2))+"%)")          

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)         
with cols[1]:
     ticker1 = "KEYRATE_D"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
     
     if is_t1 :
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         macro_data = sovdb_read(ticker1_sel, short_date)
         macro_data = macro_data.rename(columns={"Value": ticker1})         
         df = macro_data[ticker1].to_frame()
         df = df.sort_index()
         
         df_y = df.resample('Y').last()
         YTD_FX = (df_y.values[-1][0]/df_y.values[-2][0]-1)*100
         
         ax.plot(df, color=mymap[0], linewidth=0.8)          
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#         
         ax.plot(df_y.index[-2], df_y.values[-2][0], marker=5,color=(1,0,0)) 
                       
         End_val = df.values[-1][0]
         Start_val = df_y.values[-2][0]

         period_ret = (End_val-Start_val)*100       
                         
         plt.title("Key rate "+str(df.index[-1].strftime('%Y-%m-%d'))+", YTD:"+str(round(period_ret,2))+"bp")          

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)          

cols=st.columns(2)
with cols[0]:
     ticker1 = "STOCKMARKETLC_D"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel) 
     
     if is_t1 :
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         macro_data = sovdb_read(ticker1_sel, short_date)
         macro_data = macro_data.rename(columns={"Value": ticker1})         
         df = macro_data[ticker1].to_frame()
         
         df_y = df.resample('Y').last()
         YTD_FX = (df_y.values[-1][0]/df_y.values[-2][0]-1)*100
         
         ax.plot(df, color=mymap[0], linewidth=0.8)          
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#         
         ax.plot(df_y.index[-2], df_y.values[-2][0], marker=5,color=(1,0,0)) 
                       
         End_val = df.values[-1][0]
         Start_val = df_y.values[-2][0]
         End_date_c = df.index[-1]
         Start_date_c = df_y.index[-2]

         period_ret = (End_val/Start_val-1)*100
         annula_ret = ((1+period_ret/100)**(365.25/(End_date_c - Start_date_c).days)-1)*100
         years = (End_date_c - Start_date_c).days/365.25       
                         
         plt.title("Stock market in LC "+str(df.index[-1].strftime('%Y-%m-%d'))+", YTD:"+str(round(YTD_FX,2))+"% (ann:"+str(round(annula_ret,2))+"%)")          

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig) 
         
st.subheader('Macro')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "NGDP_RPCH_Y_WEO"
     ticker1_sel = key+"_"+ticker1     
     is_t1 = ticker_exists(ticker1_sel)     
     
     ticker2 = "PCPIPCH_Y_WEO"
     ticker2_sel = key+"_"+ticker2     
     is_t2 = ticker_exists(ticker2_sel)    
     
     if is_t1 and is_t2:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #indicator1
         macro_data = sovdb_read(ticker1_sel, st_date)
         macro_data = macro_data.rename(columns={"Value": ticker1})         
         df = macro_data[ticker1].to_frame()
                  
         end_window = 2019
         window_step = 10
         gdp_g_last = df.loc[datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'):datetime(end_window, 12, 31).strftime('%Y-%m-%d')].mean()[0]         
         df_g = pd.DataFrame(np.repeat(gdp_g_last,10), index=pd.date_range(start=datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'), end=datetime(end_window, 12, 31).strftime('%Y-%m-%d'), freq='Y'))         
         
         ax.plot(df, color=mymap[0], label='gdp growth',linewidth=0.8) 
         ax.plot(df_g, color=mymap[0], linewidth=0.8) 
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
         ax.text(df_g.index[-1], df_g.values[-1][0], round(df_g.values[-1][0],2), fontsize=8,color=mymap[0])#
         
         #indicator2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame()
         
         cpi_last = df.loc[datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'):datetime(end_window, 12, 31).strftime('%Y-%m-%d')].mean()[0]         
         df_c = pd.DataFrame(np.repeat(cpi_last,10), index=pd.date_range(start=datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'), end=datetime(end_window, 12, 31).strftime('%Y-%m-%d'), freq='Y'))         
         
         ax.plot(df, color=mymap[1], label='cpi, avg',linewidth=0.8) 
         ax.plot(df_c, color=mymap[1], linewidth=0.8) 
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[1])#
         ax.text(df_c.index[-1], df_c.values[-1][0], round(df_c.values[-1][0],2), fontsize=8,color=mymap[1])#
         
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)        
         
         if limit_y:
             mult1 = 3
             mult2 = 9
             y_lim_up = min(macro_data[ticker1].median()*mult2,macro_data[ticker2].median()*mult1)
             y_lim_down = max(macro_data[ticker1].median()*(-2),macro_data[ticker2].median()*(-1))
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
     is_t1 = ticker_exists(ticker1_sel)  
     
     ticker2 = "NGDPD_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)      
     
     if is_t1 and is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame()         
         p1 = ax.plot(df, color=mymap[0],  label='gdp, constant',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')

         #ticker2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame()
        
         ax2 = ax.twinx()
         p2 = ax2.plot(df, color=mymap[1], label='gdp, bln USD, rhs',linewidth=0.8)               
             
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
     is_t1 = ticker_exists(ticker1_sel)  
     
     ticker2 = "DDFERTRATE_Y_CUST"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)  
     
     if is_t1:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame() 
         
         p1 = ax.plot(df, color=mymap[0], label='population',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')         
         pop_last = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         pop_10Y = df.loc[datetime(date.today().year-10, 12, 31).strftime('%Y-%m-%d')]
         pop_pch = ((pop_last.values[0]/pop_10Y.values[0])**(1/10)-1)*100         
         ax.text(datetime(date.today().year-1, 12, 31), pop_last, "10Y: "+str(round(pop_pch,1))+"%", fontsize=8,color='r');
         
         #ticker2        
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame()        
         
         ax2 = ax.twinx()
         p2 = ax2.plot(df, color=mymap[1], label='fertility rate, rhs',linewidth=0.8) 
             
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
     is_t1 = ticker_exists(ticker1_sel)
     
     ticker2 = "NGDPDPC_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel)      
         
     if is_t1:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame() 
         
         p1 = ax.plot(df, color=mymap[0], label='PPP, internationl $',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         last1 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         ax.text(datetime(date.today().year-1, 12, 31), last1, last1.values[0], fontsize=8,color=mymap[0]);
        
         #ticker2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame()          
         
         ax2 = ax.twinx()
         p2 = ax2.plot(df, color=mymap[1], label='USD, rhs',linewidth=0.8) 
         last2 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
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
     is_t1 = ticker_exists(ticker1_sel)
     
     ticker2 = "GGX_NGDP_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     is_t2 = ticker_exists(ticker2_sel) 
    
     
     if is_t1 and is_t2:
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame() 
         
         ax.plot(df, color=mymap[0], label='revenues',linewidth=0.8) 
         
         #ticker2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame()
         
         ax.plot(df, color=mymap[1], label='expenditures',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
             
         plt.title("GG revenues vs expenditures, %GDP") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)

is_tp2 = 0         
with cols[1]:
     ticker1 = "GGXCNL_NGDP_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)
     
     ticker2 = "GGXONLB_NGDP_Y_WEO"
     ticker2_sel = key+"_"+ticker2
     is_tp2 = ticker_exists(ticker2_sel)      
     
     if is_t1 and is_tp2:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame().dropna()        
         ax.plot(df, color=mymap[0], label='balance',linewidth=0.8)         
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
         
         end_window = 2019
         window_step = 5
         bal_last = df.loc[datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'):datetime(end_window, 12, 31).strftime('%Y-%m-%d')].mean()[0]         
         df_bal = pd.DataFrame(np.repeat(bal_last,window_step), index=pd.date_range(start=datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'), end=datetime(end_window, 12, 31).strftime('%Y-%m-%d'), freq='Y'))         
         
         ax.plot(df_bal, color=mymap[0], linewidth=0.8)          
         ax.text(df_bal.index[-1], df_bal.values[-1][0], round(df_bal.values[-1][0],2), fontsize=8,color=mymap[0])#
                  
         #ticker2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker2].to_frame().dropna()         
                  
         ax.plot(df, color=mymap[1], label='primary balance',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
           
         plt.title("GG balances, %GDP") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)

cols=st.columns(2)        
with cols[0]:
    ticker1 = "GGXWDG_NGDP_Y_WEO"    
    ticker1_sel = key+"_"+ticker1
    is_t1 = ticker_exists(ticker1_sel)
  
    if is_t1:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        
        #ticker1
        temp = sovdb_read(ticker1_sel, st_date)
        temp = temp.rename(columns={"Value": ticker1})         
        macro_data = macro_data.join(temp, how="outer")
        df = macro_data[ticker1].to_frame().dropna()    
        
        p1 = ax.plot(df, color=mymap[0], label='% GDP',linewidth=0.8) 
        ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
        last1 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
        ax.text(datetime(date.today().year-1, 12, 31), last1, last1.values[0], fontsize=8,color=mymap[0]);
        
        macro_data['DEBT_REV'] = macro_data['GGXWDG_NGDP_Y_WEO']/macro_data['GGR_NGDP_Y_WEO']*100
        df = macro_data['DEBT_REV'].to_frame().dropna() 
        
        ax2 = ax.twinx()
        p2 = ax2.plot(df, color=mymap[1], label='% REV, rhs',linewidth=0.8) 
        last2 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
        ax2.text(datetime(date.today().year-1, 12, 31), last2, round(last2.values[0],1), fontsize=8,color=mymap[1]);
        
        plt.title("GG Debt") 
        p12 = p1+p2
        labs = [l.get_label() for l in p12]
        ax.legend(p12, labs, loc=3, frameon=False)
    
        formatter = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(formatter)
        plt.show() 
        st.pyplot(fig)
 
with cols[1]:     
   
     if is_tp2:
         macro_data['INT_GDP'] = macro_data['GGXONLB_NGDP_Y_WEO']-macro_data['GGXCNL_NGDP_Y_WEO']
         macro_data['INT_REV'] = macro_data['INT_GDP']/macro_data['GGR_NGDP_Y_WEO']*100
         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1                  
         df = macro_data['INT_GDP'].to_frame().dropna()    
         
         p1 = ax.plot(df, color=mymap[0], label='% GDP',linewidth=0.8) 
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         last1 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         ax.text(datetime(date.today().year-1, 12, 31), last1, round(last1.values[0],1), fontsize=8,color=mymap[0]);
         
         df = macro_data['INT_REV'].to_frame().dropna() 
         
         ax2 = ax.twinx()
         p2 = ax2.plot(df, color=mymap[1], label='% REV, rhs',linewidth=0.8) 
         last2 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         ax2.text(datetime(date.today().year-1, 12, 31), last2, round(last2.values[0],1), fontsize=8,color=mymap[1]);
         
         plt.title("GG interest") 
         p12 = p1+p2
         labs = [l.get_label() for l in p12]
         ax.legend(p12, labs, loc=3, frameon=False)
     
         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)
         
         
st.subheader('External')
cols=st.columns(2)        
with cols[0]:
     ticker1 = "BCA_NGDPD_Y_WEO"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)
     
     if is_t1:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame().dropna()   
                  
         ax.plot(df, color=mymap[0], label='current account',linewidth=0.8)          
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
         ax.axhline(y=0, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)              
             
         end_window = 2019
         window_step = 5
         ca_last = df.loc[datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'):datetime(end_window, 12, 31).strftime('%Y-%m-%d')].mean()[0]         
         df_ca = pd.DataFrame(np.repeat(ca_last,window_step), index=pd.date_range(start=datetime(end_window-window_step+1, 12, 31).strftime('%Y-%m-%d'), end=datetime(end_window, 12, 31).strftime('%Y-%m-%d'), freq='Y'))         
              
         ax.plot(df_ca, color=mymap[0], linewidth=0.8)          
         ax.text(df_ca.index[-1], df_ca.values[-1][0], round(df_ca.values[-1][0],2), fontsize=8,color=mymap[0])#
         
         plt.title("Current account balance, %GDP")         

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)         
with cols[1]:
     ticker1 = "DDUSDLCAVGT_Y_CUST"
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)
     
     if is_t1:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         temp = sovdb_read(ticker1_sel, st_date)
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         df = macro_data[ticker1].to_frame().dropna()   
                               
         p1 = ax.plot(df, color=mymap[0], label='LCUSD',linewidth=0.8)          
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
         #ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')                     
         
         df_yoy = df.pct_change(periods=1) * 100 
         #st.write(df_yoy)
         df_yoy = df_yoy.rename(columns={"DDUSDLCAVGT_Y_CUST": "LCUSD_YOY"}) 
         macro_data = macro_data.join(df_yoy, how="outer")
         df = df_yoy
         
         ax2 = ax.twinx()
         p2 = ax2.plot(df, 'o', color=mymap[1], label='%yoy, rhs',linewidth=0.8) 
         ax2.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[1])#
         #last2 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
         #ax2.text(datetime(date.today().year-1, 12, 31), last2, round(last2.values[0],1), fontsize=8,color=mymap[1]);
         
         plt.title("Local currency to USD")         
         p12 = p1+p2
         labs = [l.get_label() for l in p12]
         ax.legend(p12, labs, loc=3, frameon=False)

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)     

cols=st.columns(2) 
with cols[0]:
     ticker1 = "DDEXTDUSD_Y_WB" #External Debt USD
     ticker1_sel = key+"_"+ticker1
     is_t1 = ticker_exists(ticker1_sel)
     
     ticker2 = "DDNIIPRESGDP_Y_CUST" #Reserves, %GDP NGDPD_Y_WEO
     ticker2_sel = key+"_"+ticker2
     is_tp2 = ticker_exists(ticker2_sel)      
     
     if is_t1 and is_tp2:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)
         
         #ticker1
         temp = sovdb_read(ticker1_sel, st_date)/1000000000
         temp = temp.rename(columns={"Value": ticker1})         
         macro_data = macro_data.join(temp, how="outer")
         macro_data['EXTD_GDP'] = macro_data['DDEXTDUSD_Y_WB']/macro_data['NGDPD_Y_WEO']*100
         df = macro_data['DDEXTDUSD_Y_WB'].to_frame().dropna()      
                  
         ax.plot(df, color=mymap[0], label='External Debt',linewidth=0.8)         
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
                           
         #ticker2
         temp = sovdb_read(ticker2_sel, st_date)
         temp = temp.rename(columns={"Value": ticker2})         
         macro_data = macro_data.join(temp, how="outer")
         macro_data['RES_USD'] = macro_data['DDNIIPRESGDP_Y_CUST']*macro_data['NGDPD_Y_WEO']/100
         df = macro_data['RES_USD'].to_frame().dropna()         
                  
         ax.plot(df, color=mymap[1], label='Reserves',linewidth=0.8) 
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[1])#
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')         
           
         plt.title("External Debt & Reserves, bln USD") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)

with cols[1]:     
     if is_t1 and is_tp2:         
         fig = plt.figure()
         ax = fig.add_subplot(1, 1, 1)

         df = macro_data['EXTD_GDP'].to_frame().dropna()                        
         ax.plot(df, color=mymap[0], label='External Debt',linewidth=0.8)         
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[0])#
                           
         #ticker2
         df = macro_data['DDNIIPRESGDP_Y_CUST'].to_frame().dropna()         
                  
         ax.plot(df, color=mymap[1], label='Reserves',linewidth=0.8) 
         ax.text(df.index[-1], df.values[-1][0], round(df.values[-1][0],2), fontsize=8,color=mymap[1])#
         ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')         
           
         plt.title("External Debt & Reserves, %GDP") 
         plt.legend(loc=0,frameon=False) 

         formatter = matplotlib.dates.DateFormatter('%Y')
         ax.xaxis.set_major_formatter(formatter)
         plt.show() 
         st.pyplot(fig)        

#st.write(type(macro_data))
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
    macro_data.to_excel(writer, sheet_name='Sheet1', index=True)    
download = st.download_button(
    label="Excel",
    data=buffer,
    file_name=countr+"_macro.xlsx",
    mime='application/vnd.ms-excel'
)