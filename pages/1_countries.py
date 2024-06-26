import psycopg2 as ps
import pandas as pd
#import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import date, datetime
import io
import streamlit as st
#from streamlit_timeline import st_timeline

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
     
def get_rank(ticker_r1,key, date_rank,small_peers):
    ticker_r10 = key+"_"+ticker_r1
    ticker_c_val = sovdb_read_date(ticker_r10, date_rank)
    
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
            #st.write(ticker_x)                 
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
    
    return ticker_c_val, rank1_all, rank1_small, len(all_peers_keys), len(small_peers_key)


#read all countries des
df_all = sovdb_read_gen("countries")
count_sel = df_all.name
df_all = df_all.fillna('') 

countr = st.selectbox("Country",(count_sel), index=203)
key = df_all[df_all.name==countr].m_key.values[0]

#get all peers
df_p = sovdb_read_gen("peers")
peers = df_p.p_key

#get ratings   
df_rscale = sovdb_read_gen("Rating_Scales")
r_categories_s = df_rscale.score_cat.unique()
r_categories = df_rscale.Category.unique()
is_rating = 0

if table_exists(key+"_RATINGS"):        
    df_ratings = sovdb_read_gen(key+"_RATINGS")        
    Moodys_r = df_ratings.Moodys_r.values[-1]
    Moodys_o = df_ratings.Moodys_o.values[-1]
    SNP_r = df_ratings.SNP_r.values[-1]
    SNP_o = df_ratings.SNP_o.values[-1]
    Fitch_r = df_ratings.Fitch_r.values[-1]
    Fitch_o = df_ratings.Fitch_o.values[-1]
    
    Big3_cat_s = df_ratings.big3_cat_s.values[-1]
    Big3_cat   = df_ratings.big3_cat.values[-1]    
    Big3_cat_ind = r_categories_s[r_categories_s==Big3_cat_s]
    
    
    is_rating = 1
else:
    Moodys_r = "None"
    Moodys_o = ""
    SNP_r = "None"
    SNP_o = ""
    Fitch_r = "None"
    Fitch_o = ""

st.header('Description data')
cols=st.columns(4)

with cols[0]:
    if (len(df_all[df_all.name==countr].name.values[0])==0):
        st.write("Name: none")
    else:
        st.write("Name: "+df_all[df_all.name==countr].name.values[0])
with cols[1]:
    if (len(df_all[df_all.name==countr].pol_type.values[0])==0):
        st.write("Political system: none")
    else:
        st.write("Political system: "+df_all[df_all.name==countr].pol_type.values[0])
with cols[2]:
    if (len(df_all[df_all.name==countr].president.values[0])==0):
        st.write("President: none")
    else:
        st.write("President: "+df_all[df_all.name==countr].president.values[0])
with cols[3]:
    if (len(df_all[df_all.name==countr].type.values[0])==0):
        st.write("Type: none")
    else:
        st.write("Type: "+df_all[df_all.name==countr].type.values[0])

cols=st.columns(4)
with cols[0]:
    if (len(df_all[df_all.name==countr].fx.values[0])==0):
        st.write("Currency: none")
    else:
        st.write("Currency: "+df_all[df_all.name==countr].fx.values[0])    
with cols[1]:
    if (len(df_all[df_all.name==countr].ara_type.values[0])==0):
        st.write("IMF FX type: none")
    else:
        st.write("IMF FX type: "+df_all[df_all.name==countr].ara_type.values[0])    
with cols[2]:
    if (len(df_all[df_all.name==countr].ara_cat.values[0])==0):
        st.write("IMF FX category: none")
    else:
        st.write("IMF FX category: "+df_all[df_all.name==countr].ara_cat.values[0])          
        
cols=st.columns(4)
with cols[0]:
    st.write("Moody's: "+Moodys_r+" "+Moodys_o)       
with cols[1]:
    st.write("S&P's: "+SNP_r+" "+SNP_o)     
with cols[2]:
    st.write("Fitch's: "+Fitch_r+" "+Fitch_o) 
    
cols=st.columns(7)
with cols[0]:
    if (len(df_all[df_all.name==countr].mof_page.values[0])==0):
        st.write("MoF: none")
    else:      
        st.write("MoF: [link]("+df_all[df_all.name==countr].mof_page.values[0]+")")
with cols[1]:
    if (len(df_all[df_all.name==countr].cbr_page.values[0])==0):
        st.write("CBR: none")
    else:      
        st.write("CBR: [link]("+df_all[df_all.name==countr].cbr_page.values[0]+")")
with cols[2]:
    if (len(df_all[df_all.name==countr].stat_page.values[0])==0):
        st.write("Nat stat: none")
    else:      
        st.write("Nat stat: [link]("+df_all[df_all.name==countr].stat_page.values[0]+")")
with cols[3]:
    if (len(df_all[df_all.name==countr].oth1_page.values[0])==0):
        st.write("Other: none")
    else:      
        st.write("Other: [link]("+df_all[df_all.name==countr].oth1_page.values[0]+")")

#cols=st.columns(4)
with cols[4]:
    if (len(df_all[df_all.name==countr].imf_page.values[0])==0):
        st.write("IMF: none")
    else:      
        st.write("IMF: [link]("+df_all[df_all.name==countr].imf_page.values[0]+")")
with cols[5]:
    if (len(df_all[df_all.name==countr].wb_page.values[0])==0):
        st.write("WB: none")
    else:      
        st.write("WB: [link]("+df_all[df_all.name==countr].wb_page.values[0]+")")
with cols[6]:
    if (len(df_all[df_all.name==countr].ebrd_page.values[0])==0):
        st.write("EBRD: none")
    else:      
        st.write("EBRD: [link]("+df_all[df_all.name==countr].ebrd_page.values[0]+")")

        
cols=st.columns(8)
with cols[0]:
    st.write("Master key: "+key)
with cols[1]:
    if (len(df_all[df_all.name==countr].imf_key.values[0])==0):
        st.write("IMF key: none")
    else:
        st.write("IMF key: "+df_all[df_all.name==countr].imf_key.values[0])
with cols[2]:
    if (len(df_all[df_all.name==countr].wb_key.values[0])==0):
        st.write("WB key: none")
    else:
        st.write("WB key: "+df_all[df_all.name==countr].wb_key.values[0])      
with cols[3]:    
    if (len(df_all[df_all.name==countr].ecb_key.values[0])==0):
        st.write("ECB key: none")
    else:
        st.write("ECB key: "+df_all[df_all.name==countr].ecb_key.values[0])

#cols=st.columns(4)
with cols[4]:
    if (len(df_all[df_all.name==countr].oecd_key.values[0])==0):
        st.write("OECD key: none")
    else:
        st.write("OECD key: "+df_all[df_all.name==countr].oecd_key.values[0])
with cols[5]:
    if (len(df_all[df_all.name==countr].bis_key.values[0])==0):
        st.write("BIS key: none")
    else:
        st.write("BIS key: "+df_all[df_all.name==countr].bis_key.values[0])
with cols[6]:
    if (len(df_all[df_all.name==countr].iso2_key.values[0])==0):
        st.write("ISO2 key: none")
    else:
        st.write("ISO2 key: "+df_all[df_all.name==countr].iso2_key.values[0])      
with cols[7]:    
    if (len(df_all[df_all.name==countr].iso3_key.values[0])==0):
        st.write("ISO3 key: none")
    else:
        st.write("ISO3 key: "+df_all[df_all.name==countr].iso3_key.values[0])


#member of
member_of = []    
for peer in peers:
    peer_tick = "PP_"+peer
    df_mp = sovdb_read_gen(peer_tick)
    peers_t = df_mp.m_key
    if key in peers_t.to_list():
        member_of.append(peer)
st.write("Member of: "+', '.join(member_of))
    
st.header('Rankings')
cols=st.columns(3) 
with cols[0]:
    date_rank = st.date_input("as of: ", pd.to_datetime('2022-12-31'))
with cols[1]:
    small_peers = st.selectbox("Peers",peers, index=0)
with cols[2]:
    calc_rank = st.checkbox('Calc rankings',0)    
    
if calc_rank:
    ticker_r0 = 'NGDPD_Y_WEO'
    last0, rank0_all, rank0_small, l_n, s_n = get_rank(ticker_r0, key, date_rank, small_peers)    
    st.write("GDP: "+str(round(last0,1))+" bln USD, #"+str(rank0_all)+"/"+str(l_n)+" (peers #"+str(rank0_small)+"/"+str(s_n)+")")
    
    ticker_r1 = 'LP_Y_WEO'
    last1, rank1_all, rank1_small, l_n, s_n = get_rank(ticker_r1, key, date_rank, small_peers)    
    st.write("Population: "+str(round(last1,1))+" mln, #"+str(rank1_all)+"/"+str(l_n)+" (peers #"+str(rank1_small)+"/"+str(s_n)+")")

    ticker_r2 = 'NGDPDPC_Y_WEO'
    last2, rank2_all, rank2_small, l_n, s_n = get_rank(ticker_r2, key, date_rank, small_peers)    
    st.write("GDP per Capita: "+str(round(last2,1))+" USD, #"+str(rank2_all)+"/"+str(l_n)+" (peers #"+str(rank2_small)+"/"+str(s_n)+")")
    
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
         
st.subheader('Ratings')
if is_rating:
    cols=st.columns(2)        
    with cols[0]:    
        df_ratings = df_ratings[df_ratings.Date>=st_date]        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        #df_ratings
        ax.plot(df_ratings.Date, df_ratings.Moodys_s, color=mymap[0], label='Moodys',linewidth=0.8) 
        ax.plot(df_ratings.Date, df_ratings.SNP_s, color=mymap[1], label='S&P',linewidth=0.8) 
        ax.plot(df_ratings.Date, df_ratings.Fitch_s, color=mymap[2], label='Fitch',linewidth=0.8) 
        
        max_y = np.min(df_ratings.Moodys_s.dropna().to_list() + df_ratings.SNP_s.dropna().to_list()+ df_ratings.Fitch_s.dropna().to_list())-1
        min_y = np.max(df_ratings.Moodys_s.dropna().to_list() + df_ratings.SNP_s.dropna().to_list()+ df_ratings.Fitch_s.dropna().to_list())+1    
        
        plt.gca().invert_yaxis()    
        plt.yticks(range(1,len(df_rscale.Generic)+1), df_rscale.Generic)  # Set text labels.    
        plt.ylim(min_y, max_y)    
        
        plt.title("BIG3") 
        plt.legend(loc=0,frameon=False) 
        formatter = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(formatter)
        plt.show() 
        st.pyplot(fig)
             
    with cols[1]:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)        
        ax.plot(df_ratings.Date, df_ratings.big3_s, color=mymap[0], label='Moodys',linewidth=0.8)     
        plt.gca().invert_yaxis()
        plt.yticks(range(1,len(df_rscale.Generic)+1), df_rscale.Generic)  # Set text labels.    
        plt.title("Average BIG3") 
        plt.ylim(min_y, max_y)
        
        formatter = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(formatter)
        plt.show() 
        st.pyplot(fig)

vs_peers = st.checkbox('Macro vs rating peers',0) 
if vs_peers:
    cols=st.columns(4)
    with cols[0]:
        by_rating = st.checkbox('by rating category',1) 
    with cols[1]:
        selected_category = st.selectbox("Category: ",pd.DataFrame(r_categories), index=int(Big3_cat_ind[0])-1)
    with cols[2]:
        by_peer = st.checkbox('by peer group',0) 
    with cols[3]:
        selected_peers = st.selectbox("by peers",peers, index=0)
    
    if is_rating:
        # 
        peers_cat_s = []
        peers_key_sm = "PP_WEO"
        df = sovdb_read_gen(peers_key_sm)        
        cntr = df.country.to_list()
        peers_key = df.m_key 
        r_peers_key = []
        selected_category_s = r_categories_s[r_categories==selected_category]       
        
        i = 0
        if by_rating:
            vs_peers_name = selected_category            
            for peer in peers_key:
                ticker = peer+"_RATINGS"                        
                if table_exists(ticker):
                    df_ratings = sovdb_read_gen(ticker)                     
                    if selected_category_s == df_ratings.big3_cat_s.values[-1]:                        
                        r_peers_key.append(peer)
                        peers_cat_s.append(cntr[i] +"("+peer+")")
                i+=1
            
        if by_peer:
            peers_cat_s = []
            r_peers_key = []
            vs_peers_name = selected_peers 
            
            peers_key_selected = "PP_"+selected_peers
            df = sovdb_read_gen(peers_key_selected)        
            cntr = df.country.to_list()
            r_peers_key = df.m_key.to_list() 
            i=0
            for peer in r_peers_key:                            
                peers_cat_s.append(cntr[i] +"("+peer+")")               
                i+=1
                
        if key not in(r_peers_key):            
            r_peers_key.append(key)
            peers_cat_s.append(countr +"("+key+")")
            
        date_p = st.date_input("As of: ", pd.to_datetime('2022-12-31'))
                
        GDP_p = []         
        Pop_p = [] 
        GDP_PPP_USD_p = [] 
        GDP_g_p = [] 
        CPI_p = [] 
        
        GGBAL_GDP_p = []
        GGPBAL_GDP_p = []
        GGDEBT_GDP = []
        GGR_GDP = []
        GGE_GDP = []
        GGD_REV = []
        GGINT_GDP = []
        GGINT_REV = []
        
        CA_GDP_p = []
        EXTD_GDP_p = []
        RES_USD_p = []
        RES_GDP_p = []
        
        years_shift = 10
        peers_d_p = datetime(date_p.year-years_shift, date_p.month, date_p.day)    
        
        years_shift_p = 5
        peers_d_pp = datetime(date_p.year-years_shift_p, date_p.month, date_p.day)    
        
        for peer in r_peers_key:
            #GDP                
            ticker_g = peer+"_NGDPD_Y_WEO"            
            if (ticker_exists(ticker_g)):
                GDP_p.append(sovdb_read_date(ticker_g,date_p))
            #Pop                
            ticker_p = peer+"_LP_Y_WEO"            
            if (ticker_exists(ticker_p)):
                Pop_p.append(sovdb_read_date(ticker_p,date_p))
            
            #GDP USD per Capita                
            ticker_pc = peer+"_NGDPDPC_Y_WEO"            
            if (ticker_exists(ticker_pc)):
                GDP_PPP_USD_p.append(sovdb_read_date(ticker_pc,date_p)/1000)
                
            #GDP growth
            ticker_g_g = peer+"_NGDP_RPCH_Y_WEO"            
            if (ticker_exists(ticker_g_g)):
                temp = sovdb_read(ticker_g_g, peers_d_p)
                GDP_g_p.append(round(temp.values[1:1+years_shift].mean(),1))
    
            #CPI
            ticker_cpi = peer+"_PCPIPCH_Y_WEO"            
            if (ticker_exists(ticker_cpi)):
                temp = sovdb_read(ticker_cpi, peers_d_pp)
                CPI_p.append(round(temp.values[1:1+years_shift_p].mean(),1))                                
                
            #GGBAL_GDP            
            ticker_ggb = peer+"_GGXCNL_NGDP_Y_WEO"            
            if (ticker_exists(ticker_ggb)):
                temp = sovdb_read(ticker_ggb, peers_d_pp)
                GGBAL_GDP_p.append(round(temp.values[1:1+years_shift_p].mean(),1))                                

            #GGPBAL_GDP            
            ticker_ggpb = peer+"_GGXONLB_NGDP_Y_WEO"            
            if (ticker_exists(ticker_ggpb)):
                temp = sovdb_read(ticker_ggpb, peers_d_pp)
                GGPBAL_GDP_p.append(round(temp.values[1:1+years_shift_p].mean(),1))                                
                
            #GGREV / GDP                
            ticker_ggr = peer+"_GGR_NGDP_Y_WEO"            
            if (ticker_exists(ticker_ggr)):
                GGR_GDP.append(sovdb_read_date(ticker_ggr,date_p))
                
            #GGEXP / GDP                
            ticker_gge = peer+"_GGX_NGDP_Y_WEO"            
            if (ticker_exists(ticker_gge)):
                GGE_GDP.append(sovdb_read_date(ticker_gge,date_p))
                
            #GGDEBT / GDP                
            ticker_ggdebt = peer+"_GGXWDG_NGDP_Y_WEO"            
            if (ticker_exists(ticker_ggdebt)):
                GGDEBT_GDP.append(sovdb_read_date(ticker_ggdebt,date_p))
            
            #CA_GDP            
            ticker_ca = peer+"_BCA_NGDPD_Y_WEO"            
            if (ticker_exists(ticker_ca)):
                temp = sovdb_read(ticker_ca, peers_d_pp)
                CA_GDP_p.append(round(temp.values[1:1+years_shift_p].mean(),1))      
                
            #External debt, %GDP
            ticker_extd = peer+"_DDNIIPRESGDP_Y_CUST"            
            if (ticker_exists(ticker_extd)):
                EXTD_GDP_p.append(sovdb_read_date(ticker_extd,date_p)) 
            else:
                #st.write(peer)
                EXTD_GDP_p.append(0)
  
            #Reserves, USD
            ticker_res = peer+"_DDEXTDUSD_Y_WB"            
            if (ticker_exists(ticker_res)):
                RES_USD_p.append(sovdb_read_date(ticker_res,date_p)) 
            else:
                #st.write(peer)
                RES_USD_p.append(0)
                    
        #GGDEBT / REV 
        GGD_REV = [m/n*100 for m, n in zip(GGDEBT_GDP, GGR_GDP)]
       
        #INT / GDP        
        GGINT_GDP = [m - n for m, n in zip(GGPBAL_GDP_p, GGBAL_GDP_p)]
        
        #INT / REV
        GGINT_REV = [m/n*100 for m, n in zip(GGINT_GDP, GGR_GDP)]
                
        ticker_p_c = key+"_NGDPD_Y_WEO"
        GDP_p_c = sovdb_read_date(ticker_p_c,date_p)        
        
        ticker_p_c = key+"_LP_Y_WEO"
        Pop_p_c = sovdb_read_date(ticker_p_c,date_p)        
        
        ticker_pc_c = key+"_NGDPDPC_Y_WEO"
        GDP_PPP_USD_p_c = sovdb_read_date(ticker_pc_c,date_p)/1000
        
        ticker_g_g = key+"_NGDP_RPCH_Y_WEO"
        temp = sovdb_read(ticker_g_g, peers_d_p)
        GDP_g_p_c = round(temp.values[1:1+years_shift].mean(),1)
        
        ticker_cpi = key+"_PCPIPCH_Y_WEO"
        temp = sovdb_read(ticker_cpi, peers_d_pp)
        CPI_p_c = round(temp.values[1:1+years_shift_p].mean(),1)        
        
        #FISCAL
        #GG Bal
        ticker_ggb = key+"_GGXCNL_NGDP_Y_WEO"
        temp = sovdb_read(ticker_ggb, peers_d_pp)
        GGBAL_GDP_p_c = round(temp.values[1:1+years_shift_p].mean(),1)
        
        #GG Primary Bal
        ticker_ggpb = key+"_GGXONLB_NGDP_Y_WEO"
        temp = sovdb_read(ticker_ggpb, peers_d_pp)
        GGPBAL_GDP_p_c = round(temp.values[1:1+years_shift_p].mean(),1)
        
        #GG rev
        ticker_ggr = key+"_GGR_NGDP_Y_WEO"
        temp = sovdb_read_date(ticker_ggr, peers_d_p)
        GGR_GDP_c = sovdb_read_date(ticker_ggr,date_p)
        
        #GG exp
        ticker_gge = key+"_GGX_NGDP_Y_WEO"
        temp = sovdb_read_date(ticker_gge, peers_d_p)
        GGE_GDP_c = sovdb_read_date(ticker_gge,date_p)
        
        #GG Debt
        ticker_ggdebt = key+"_GGXWDG_NGDP_Y_WEO"
        temp = sovdb_read_date(ticker_ggdebt, peers_d_p)
        GGDEBT_GDP_c = sovdb_read_date(ticker_ggdebt,date_p)
        
        #GGDebt / REV
        GGD_REV_c = GGDEBT_GDP_c / GGR_GDP_c*100
        
        #GGINT / GDP
        GGINT_GDP_c = GGPBAL_GDP_p_c - GGBAL_GDP_p_c
        
        #GGINT / REV
        GGINT_REV_c = GGINT_GDP_c / GGR_GDP_c*100
        
        #EXTERNAL
        #Current account, %GDP
        ticker_ca = key+"_BCA_NGDPD_Y_WEO"
        temp = sovdb_read(ticker_ca, peers_d_pp)
        CA_GDP_p_c = round(temp.values[1:1+years_shift_p].mean(),1)
        
        #External debt, %GDP
        ticker_extd = key+"_DDNIIPRESGDP_Y_CUST"
        temp = sovdb_read_date(ticker_extd, peers_d_p)
        EXTD_GDP_p_c = sovdb_read_date(ticker_extd,date_p)
        
        #RES / GDP
        RES_GDP_p = [(m/1000000000)/n*100 for m, n in zip(RES_USD_p, GDP_p)]        
        
        ticker_res = key+"_DDEXTDUSD_Y_WB"
        temp = sovdb_read_date(ticker_res, peers_d_p)
        RES_USD_p_c = sovdb_read_date(ticker_res,date_p)
        
        RES_GDP_p_c = (RES_USD_p_c/1000000000) / GDP_p_c *100
        
        #add to dataframe        
        df_f = pd.DataFrame({'GDP_USD':GDP_p, 'POP':Pop_p, 'GDP_pc_USD':GDP_PPP_USD_p,'GDP_g_10Y':GDP_g_p,'CPI_5Y':CPI_p,\
                             'CA_5Y':CA_GDP_p, 'EXTD_GDP':EXTD_GDP_p, 'RES_GDP':RES_GDP_p,\
                             'GGBAL_GDP_5Y':GGBAL_GDP_p, 'GGPBAL_GDP_5Y':GGPBAL_GDP_p, 'REV_GDP':GGR_GDP, 'EXP_GDP':GGE_GDP,\
                             'GGDEBT_GDP':GGDEBT_GDP, 'GGDEBT_REV':GGD_REV, 'INT_GDP':GGINT_GDP, 'INT_REV':GGINT_REV}, index=r_peers_key)
        
        #excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
            df_f.to_excel(writer, sheet_name='Sheet1', index=True)    
        
        download2 = st.download_button(
            label="Excel",
            data=buffer,
            file_name=countr+"-"+Big3_cat+".xlsx",
            mime='application/vnd.ms-excel'
        ) 
       
        cols=st.columns(2)        
        with cols[0]:
            ###MACRO
            fig = plt.figure()
            ax = fig.add_subplot(1, 4, 1)
            #ax = fig.add_subplot(nrows=1, ncols=3, figsize=(6, 6), sharey=True)
            ax.boxplot(GDP_p,labels=['GDP, bln USD'])
            for i in range(len(GDP_p)):
                if GDP_p[i] == GDP_p_c:
                    ax.plot(1, GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GDP_p[i], marker='.', color=mymap[0], alpha=0.4)
    
           # ax = fig.add_subplot(1, 5, 2)        
           # ax.boxplot(Pop_p,labels=['Popul, mln'])
           # for i in range(len(Pop_p)):
           #     if Pop_p[i] == Pop_p_c:
           #         ax.plot(1, Pop_p[i], marker='x', color='r', alpha=0.9)
           #     else:
           #         x = np.random.normal(1, 0.04, size=1)
           #         ax.plot(x, Pop_p[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 2)        
            ax.boxplot(GDP_PPP_USD_p,labels=['GDP pc, 000 USD'])
            for i in range(len(GDP_PPP_USD_p)):
                if GDP_PPP_USD_p[i] == GDP_PPP_USD_p_c:
                    ax.plot(1, GDP_PPP_USD_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GDP_PPP_USD_p[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 3)        
            ax.boxplot(GDP_g_p,labels=['GDP, 10Yg'])
            for i in range(len(GDP_g_p)):
                if GDP_g_p[i] == GDP_g_p_c:
                    ax.plot(1, GDP_g_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GDP_g_p[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 4)        
            ax.boxplot(CPI_p,labels=['CPI, 5Yg'])
            for i in range(len(CPI_p)):
                if CPI_p[i] == CPI_p_c:
                    ax.plot(1, CPI_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, CPI_p[i], marker='.', color=mymap[0], alpha=0.4)
                    
            plt.suptitle(countr+" ("+Big3_cat+") vs "+vs_peers_name+" peers (macro)")
            plt.show() 
            st.pyplot(fig)
        with cols[1]:
            #EXTERNAL
            fig = plt.figure()      
            
            ax = fig.add_subplot(1, 3, 1)        
            ax.boxplot(CA_GDP_p,labels=['CA, 5Y, %GDP'])
            for i in range(len(CA_GDP_p)):
                if CA_GDP_p[i] == CA_GDP_p_c:
                    ax.plot(1, CA_GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, CA_GDP_p[i], marker='.', color=mymap[0], alpha=0.4)

            ax = fig.add_subplot(1, 3, 2)        
            ax.boxplot(RES_GDP_p,labels=['Res, %GDP'])
            for i in range(len(RES_GDP_p)):
                if RES_GDP_p[i] == RES_GDP_p_c:
                    ax.plot(1, RES_GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, RES_GDP_p[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 3, 3)        
            ax.boxplot(EXTD_GDP_p,labels=['ExtD, %GDP'])
            for i in range(len(EXTD_GDP_p)):
                if EXTD_GDP_p[i] == EXTD_GDP_p_c:
                    ax.plot(1, EXTD_GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, EXTD_GDP_p[i], marker='.', color=mymap[0], alpha=0.4)    
            plt.suptitle(countr+" ("+Big3_cat+") vs "+vs_peers_name+" peers (external)")
            plt.show() 
            st.pyplot(fig)
                    
        cols=st.columns(2)        
        with cols[0]:
            #FISCAL 1
            fig = plt.figure()
            ax = fig.add_subplot(1, 4, 1)        
            ax.boxplot(GGBAL_GDP_p,labels=['Bal, 5Y, %GDP'])
            for i in range(len(GGBAL_GDP_p)):
                if GGBAL_GDP_p[i] == GGBAL_GDP_p_c:
                    ax.plot(1, GGBAL_GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGBAL_GDP_p[i], marker='.', color=mymap[0], alpha=0.4)
    
            ax = fig.add_subplot(1, 4, 2)        
            ax.boxplot(GGPBAL_GDP_p,labels=['Pr bal, 5Y, %GDP'])
            for i in range(len(GGPBAL_GDP_p)):
                if GGPBAL_GDP_p[i] == GGPBAL_GDP_p_c:
                    ax.plot(1, GGPBAL_GDP_p[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGPBAL_GDP_p[i], marker='.', color=mymap[0], alpha=0.4)
                  
            ax = fig.add_subplot(1, 4, 3)        
            ax.boxplot(GGR_GDP,labels=['Rev, %GDP'])
            for i in range(len(GGR_GDP)):
                if GGR_GDP[i] == GGR_GDP_c:
                    ax.plot(1, GGR_GDP[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGR_GDP[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 4)        
            ax.boxplot(GGE_GDP,labels=['Exp, %GDP'])
            for i in range(len(GGE_GDP)):
                if GGE_GDP[i] == GGE_GDP_c:
                    ax.plot(1, GGE_GDP[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGE_GDP[i], marker='.', color=mymap[0], alpha=0.4)
                    
                    
            plt.suptitle(countr+" ("+Big3_cat+") vs "+vs_peers_name+" peers (fiscal 1)")
            plt.show() 
            st.pyplot(fig)
        
        with cols[1]:
            #FISCAL 2
            fig = plt.figure()      
            
            ax = fig.add_subplot(1, 4, 1)        
            ax.boxplot(GGDEBT_GDP,labels=['Debt, %GDP'])
            for i in range(len(GGDEBT_GDP)):
                if GGDEBT_GDP[i] == GGDEBT_GDP_c:
                    ax.plot(1, GGDEBT_GDP[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGDEBT_GDP[i], marker='.', color=mymap[0], alpha=0.4)
    
            ax = fig.add_subplot(1, 4, 2)        
            ax.boxplot(GGD_REV,labels=['Debt, %REV'])
            for i in range(len(GGD_REV)):
                if GGD_REV[i] == GGD_REV_c:
                    ax.plot(1, GGD_REV[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGD_REV[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 3)        
            ax.boxplot(GGINT_GDP,labels=['Int, %GDP'])
            for i in range(len(GGINT_GDP)):
                if GGINT_GDP[i] == GGINT_GDP_c:
                    ax.plot(1, GGINT_GDP[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGINT_GDP[i], marker='.', color=mymap[0], alpha=0.4)
                    
            ax = fig.add_subplot(1, 4, 4)        
            ax.boxplot(GGINT_REV,labels=['Int, %REV'])
            for i in range(len(GGINT_REV)):
                if GGINT_REV[i] == GGINT_REV_c:
                    ax.plot(1, GGINT_REV[i], marker='x', color='r', alpha=0.9)
                else:
                    x = np.random.normal(1, 0.04, size=1)
                    ax.plot(x, GGINT_REV[i], marker='.', color=mymap[0], alpha=0.4)
                    
            plt.suptitle(countr+" ("+Big3_cat+") vs "+vs_peers_name+" peers (fiscal 2)")
            plt.show() 
            st.pyplot(fig)
        st.write(vs_peers_name + " peers ("+str(len(peers_cat_s)-1)+"): "+', '.join(peers_cat_s))            

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
                  
         if (df.index[-1]>datetime(date.today().year-1, 12, 31)):
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
        if (df.index[-1]>datetime(date.today().year-1, 12, 31)):
            last1 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
            ax.text(datetime(date.today().year-1, 12, 31), last1, last1.values[0], fontsize=8,color=mymap[0]);
        
        macro_data['DEBT_REV'] = macro_data['GGXWDG_NGDP_Y_WEO']/macro_data['GGR_NGDP_Y_WEO']*100
        df = macro_data['DEBT_REV'].to_frame().dropna() 
        
        ax2 = ax.twinx()
        p2 = ax2.plot(df, color=mymap[1], label='% REV, rhs',linewidth=0.8) 
        if (df.index[-1]>datetime(date.today().year-1, 12, 31)):
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
         if (df.index[-1]>datetime(date.today().year-1, 12, 31)):
             last1 = df.loc[datetime(date.today().year-1, 12, 31).strftime('%Y-%m-%d')]
             ax.text(datetime(date.today().year-1, 12, 31), last1, round(last1.values[0],1), fontsize=8,color=mymap[0]);
         
         df = macro_data['INT_REV'].to_frame().dropna() 
         
         ax2 = ax.twinx()
         p2 = ax2.plot(df, color=mymap[1], label='% REV, rhs',linewidth=0.8) 
         if (df.index[-1]>datetime(date.today().year-1, 12, 31)):
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

## TIMELINE

# =============================================================================
# #st.set_page_config(layout="wide")
# st.subheader("EVENTS")
# table_name = key+"_EVENTS"       
# create_events_table = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+table_name+"\" ( ID SERIAL PRIMARY KEY, \"""Date\""" date, \"""tag\""" text, \"""des\""" text, \"""long_des\""" text)"    
# cur = conn.cursor()
# cur.execute(create_events_table)
# conn.commit()
# 
# events = sovdb_read_gen(table_name)
# 
# tags_events_all = []
# for event in events.tag:
#     x = event.split(',')
#     x = [a.strip() for a in x]
#     for x0 in x:
#         tags_events_all.append(x0)
# 
# tags_events = st.selectbox("choose tag: ",pd.Series(tags_events_all).unique(), index=0)
# 
# tag_ind = []
# for i in range(events.shape[0]):
#     if(tags_events in events.loc[i].tag):
#         tag_ind.append(float(events.loc[i].id))
# events_f = events[(events.id.isin(tag_ind))].sort_values(by=['Date'], ascending=True).reset_index()
# 
# items = []    
# for i in range(events_f.shape[0]):
#     items.append({"id":str(events_f.loc[i].id),"content": events_f.loc[i].des, "start": events_f.loc[i].Date.strftime('%Y-%b-%d')})   
# 
# #st.write(events_f)
# timeline = st_timeline(items, groups=[], options={}, height="300px")
# 
# st.write(events_f)
# if (timeline!=None):
#     sel_id = timeline.get('id')
#     selected_event = sovdb_read_item(table_name, 'id', sel_id)
#     
#     st.write(selected_event[1].strftime('%d-%b-%Y'))
#     st.write(selected_event[3])
#     st.write(selected_event[4])
#     
# 
# st.subheader("Add event")
# cols=st.columns(2) 
# with cols[0]:
#     event_date = st.date_input("Event date: ", pd.to_datetime('2000-12-31'))
# with cols[1]:
#     tags = st.text_input('Tags (, separate)', '')
# 
# des = st.text_input('event, title', '')   
# long_des = st.text_input('event, des', '')   
# 
# def click_button_add_event(Add_date,Add_tags,Add_event,Add_long_event):    
#     ticker = table_name
#     query = "INSERT INTO sovdb_schema.\""+ticker+"\" (\"""Date\""", \"""tag\""", \"""des\""", \"""long_des\""") VALUES ('"+Add_date.strftime('%d-%b-%Y')+"'::date, '"+Add_tags+"':: text, '"+Add_event+"'::text, '"+Add_long_event+"'::text)"
#     cur.execute(query)
#     conn.commit()
#     st.warning("New event was added")  
#     st.session_state.clicked = True
# 
# def click_button_delete_event(DEL,event_id_del):    
#     ticker = table_name
#     query = "DELETE FROM sovdb_schema.\""+ticker+"\" WHERE \"""id\"""='"+str(event_id_del)+"'"
#     cur.execute(query)
#     conn.commit()
#     st.warning("Event was deleted")  
#     st.session_state.clicked = True
#     
# cols=st.columns(3)    
# with cols[0]: 
#     st.button('Add event', on_click=click_button_add_event, args=(event_date,tags,des,long_des)) 
# with cols[1]:
#     event_id_del = st.selectbox("id to delete: ",events_f.id, index=0)
# with cols[2]:
#     st.button('Delete event', on_click=click_button_delete_event, args=(1,event_id_del)) 
# =============================================================================
    

