import streamlit as st
import pandas as pd
import psycopg2 as ps
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import io
import matplotlib
from sklearn.linear_model import LinearRegression
from numpy import inf

st.set_page_config(layout="centered")

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

def table_exists(ticker):
     #query_s = "SELECT * FROM sovdb_schema.\""+ticker+"\""
     query_s = "SELECT EXISTS (SELECT FROM pg_tables WHERE  schemaname = 'sovdb_schema' AND    tablename  = '"+ticker+"');"
     cur = conn.cursor()
     cur.execute(query_s)     
     rows = cur.fetchall()     
     return rows[0][0]
 
def ticker_exists(ticker):
     query_s = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker+"'"    
     cur = conn.cursor()
     cur.execute(query_s)
     rows = cur.fetchall()
     rows = np.array([*rows])
     return rows.size !=0

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
    if rows.size ==0:
        return 0
    else:
        return rows[0][1]
    
def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];
  
df_f = pd.DataFrame({'A' : []})

#df_macroind = sovdb_read_gen("macro_indicators")

#get all countries and its keys
df_countr = sovdb_read_gen("countries")
m_keys = df_countr.m_key
countries = df_countr.name

#get all peers
df = sovdb_read_gen("peers")
peers = df.p_key

cols=st.columns(4)
with cols[0]:
    ticker_x0 = st.text_input('indicator 1 (x)',"PPPPC_Y_WEO")
with cols[1]:
    ticker_y0 = st.text_input('indicator 2 (y)',"NGDPD_Y_WEO")
with cols[2]:
    date_st = st.date_input("Start date: ", pd.to_datetime('2023-12-31'))
with cols[3]:
    date = st.date_input("End date: ", pd.to_datetime('2023-12-31'))

cols=st.columns(2)
with cols[0]:
    country_sel = st.selectbox("Country",countries, index=203)    
    country_sel_key = df_countr[df_countr.name==country_sel]['m_key'].values[0]
with cols[1]:
    peers = st.selectbox("Peers",peers, index=0)

    
cols=st.columns(6)
with cols[0]:
    labls = st.checkbox('all peers labels',0) 
with cols[1]:
    log_x = st.checkbox('log x',0) 
with cols[2]:
    log_y = st.checkbox('log y',0) 
with cols[3]:
    y_x = st.checkbox('y=x',0) 
with cols[4]:
    y_trnd = st.checkbox('trend',0) 
with cols[5]:
    all_peers = st.selectbox("All peers",("WEO","EM","DM","All"), index=0)
    
cols=st.columns(4)
with cols[0]:
    xmin = st.number_input('x min:',format="%.1f",step=0.1)
with cols[1]:
    xmax = st.number_input('x max:',format="%.1f",step=0.1)
with cols[2]:
    ymin = st.number_input('y min',format="%.1f",step=0.1) 
with cols[3]:
    ymax = st.number_input('y max',format="%.1f",step=0.1)

    
plot_type = st.selectbox("Choose plot type",("","1. Scatter: 2 indicators 1 date (end)",\
                                             "2. Scatter: 1 indicator (x) 2 dates",\
                                             "3. Plot: 1 indicator (x) between 2 dates - peers only",\
                                             "4. Bar: 2 indicators 1 date (end) - peers only",\
                                             "5. Bar stacked: 1 indicator (x) between 2 dates",\
                                             "6. Bar: 1 indicator (x) 1 date (end)"), index=0)   
  
#get small peers keys
peers_sm_key = "PP_"+peers
df = sovdb_read_gen(peers_sm_key)
peers_s_keys = df.m_key 
data_x = []
data_y = []
labels = []
data_x_sm = []
data_y_sm = []
labels_sm = []
data_x_cn = []
data_y_cn = []
labels_cn = []    
suffix_x = ""
suffix_y = ""

temp = country_sel_key+"_"+ticker_x0    
query_s = "SELECT ""short_name"" FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+temp+"'"    
cur = conn.cursor()
cur.execute(query_s)
rows = cur.fetchall()
rows = np.array([*rows])   
indic_x_eng = rows[0][0]    
    
temp = country_sel_key+"_"+ticker_y0
query_s = "SELECT ""short_name"" FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+temp+"'"    
cur = conn.cursor()
cur.execute(query_s)
rows = cur.fetchall()
rows = np.array([*rows])   
indic_y_eng = rows[0][0] 
        
if plot_type=="1. Scatter: 2 indicators 1 date (end)":   
         
    if all_peers == "All":
        all_peers_keys = m_keys
    else:
        peers_key = "PP_"+all_peers
        df = sovdb_read_gen(peers_key)        
        all_peers_keys = df.m_key 
        
    #get broad peers
    for key in all_peers_keys:
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
       
        ticker_y = key+"_"+ticker_y0
        is_y = ticker_exists(ticker_y)               
        if is_x and is_y:                                
            
            df_x = sovdb_read(ticker_x, date)
            df_y = sovdb_read(ticker_y, date)
                        
            if ~df_x.empty and ~df_y.empty:            
                x_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
                y_down = df_y[(df_y.index == date.strftime('%Y-%m-%d'))]                
                if x_down.size == 0:
                    data_x.append(0)
                else:
                    data_x.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y.append(0)
                else:
                    data_y.append(y_down.values[0][0])
                labels.append(key)        
    
    #get small peers
    for key in peers_s_keys:
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
       
        ticker_y = key+"_"+ticker_y0
        is_y = ticker_exists(ticker_y)               
        if is_x and is_y:                                
            
            df_x = sovdb_read(ticker_x, date)
            df_y = sovdb_read(ticker_y, date)
                        
            if ~df_x.empty and ~df_y.empty:            
                x_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
                y_down = df_y[(df_y.index == date.strftime('%Y-%m-%d'))]                
                if x_down.size == 0:
                    data_x_sm.append(0)
                else:
                    data_x_sm.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y_sm.append(0)
                else:
                    data_y_sm.append(y_down.values[0][0])
                labels_sm.append(key)  
    
    #get selected countries data    
    ticker_x = country_sel_key+"_"+ticker_x0
    is_x = ticker_exists(ticker_x)
   
    ticker_y = country_sel_key+"_"+ticker_y0
    is_y = ticker_exists(ticker_y)               
    if is_x and is_y:                                
        
        df_x = sovdb_read(ticker_x, date)
        df_y = sovdb_read(ticker_y, date)
                    
        if ~df_x.empty and ~df_y.empty:            
            x_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
            y_down = df_y[(df_y.index == date.strftime('%Y-%m-%d'))]                
            if x_down.size == 0:
                data_x_cn.append(0)
            else:
                data_x_cn.append(x_down.values[0][0])
                
            if y_down.size == 0:
                data_y_cn.append(0)
            else:
                data_y_cn.append(y_down.values[0][0])
            labels_cn.append(country_sel_key)                          
    
    fig, ax = plt.subplots()           
    if log_x:
        #st.write(data_x)
        data_x = np.log(data_x)
        data_x_sm = np.log(data_x_sm)
        data_x_cn = np.log(data_x_cn)
        suffix_x = ", log"
    if log_y:
        data_y = np.log(data_y)  
        data_y_sm = np.log(data_y_sm)
        data_y_cn = np.log(data_y_cn)
        suffix_y = ", log"

    #ticker_x0        
    #x_label = indic_x_eng+suffix_x
    x_label = ticker_x0+suffix_x
    #y_label = indic_y_eng+suffix_y
    y_label = ticker_y0+suffix_y
    cols = ["Country", x_label, y_label]    
    df_f = pd.concat([pd.Series(labels,name=cols[0]), pd.Series(data_x,name=cols[1]), pd.Series(data_y,name=cols[2])],axis=1)  
    
    #plot broad peers
    ax.scatter(data_x,data_y,color=(0.45, 0.45, 0.45), s=10)
    #plot small peers
    ax.scatter(data_x_sm,data_y_sm,color=mymap[0], s=10)
    #selected country
    ax.scatter(data_x_cn,data_y_cn,color=mymap[1], s=10)
    
    if y_x:
        xpoints = ypoints = ax.get_xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='r', lw=1, scalex=False, scaley=False)        
    
    if y_trnd:        
        model = LinearRegression()        
        data_x[np.isneginf(data_x)] = 0
        model.fit(data_x.reshape(-1, 1), np.array(data_y))
        b = model.intercept_
        k = model.coef_
        
        x_new = np.arange(np.min(data_x),  np.max(data_x), 0.01)
        y_new = k*x_new+b
        ax.plot(x_new,y_new)
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
    if labls:
        for i, txt in enumerate(labels):
            #https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_alignment.html
            ax.annotate(txt, (data_x[i], data_y[i]),ha='left', va='bottom', size=8)
    for i, txt in enumerate(labels_sm):        
        ax.annotate(txt, (data_x_sm[i], data_y_sm[i]),ha='left', va='bottom', size=8)
    
    ax.annotate(labels_cn[0], (data_x_cn[0], data_y_cn[0]),ha='left', va='bottom', size=8)
        
    plt.title(country_sel+" vs "+peers+" vs "+all_peers+": "+date.strftime('%Y-%m-%d'))
    plt.show()    
    plt.ylim((ymin, ymax))
    plt.xlim((xmin, xmax))
    cols=st.columns(4)
    with cols[0]:
        #st.write(np.min(data_x))
        st.write("x min: "+str(round(np.min(data_x),2)))
    with cols[1]:
        st.write("x max: "+str(round(np.max(data_x),2)))
    with cols[2]:
        st.write("y min: "+str(round(np.min(data_y),2)))
    with cols[3]:
        st.write("y max: "+str(round(np.max(data_y),2)))
        
    st.pyplot(fig)


elif plot_type=="2. Scatter: 1 indicator (x) 2 dates":    
    if all_peers == "All":
        all_peers_keys = m_keys
    else:
        peers_key = "PP_"+all_peers
        df = sovdb_read_gen(peers_key)        
        all_peers_keys = df.m_key 
        
    #get broad peers
    for key in all_peers_keys:
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
        
        if is_x:             
            df_x = sovdb_read(ticker_x, date_st)            
                        
            if ~df_x.empty:            
                x_down = df_x[(df_x.index == date_st.strftime('%Y-%m-%d'))]
                y_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
                
                if x_down.size == 0:
                    data_x.append(0)
                else:
                    data_x.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y.append(0)
                else:
                    data_y.append(y_down.values[0][0])
                labels.append(key)
    #get small peers
    for key in peers_s_keys:
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
        
        if is_x:             
            df_x = sovdb_read(ticker_x, date_st)            
                        
            if ~df_x.empty:            
                x_down = df_x[(df_x.index == date_st.strftime('%Y-%m-%d'))]
                y_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
                
                if x_down.size == 0:
                    data_x_sm.append(0)
                else:
                    data_x_sm.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y_sm.append(0)
                else:
                    data_y_sm.append(y_down.values[0][0])
                labels_sm.append(key)
    
    #get selected countries data    
    ticker_x = country_sel_key+"_"+ticker_x0
    is_x = ticker_exists(ticker_x)   
        
    if is_x:             
        df_x = sovdb_read(ticker_x, date_st)            
                    
        if ~df_x.empty:            
            x_down = df_x[(df_x.index == date_st.strftime('%Y-%m-%d'))]
            y_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]
            
            if x_down.size == 0:
                data_x_cn.append(0)
            else:
                data_x_cn.append(x_down.values[0][0])
                
            if y_down.size == 0:
                data_y_cn.append(0)
            else:
                data_y_cn.append(y_down.values[0][0])
            labels_cn.append(country_sel_key)
                
    if log_x:
        data_x = np.log(data_x)
        data_x_sm = np.log(data_x_sm)
        data_x_cn = np.log(data_x_cn)
        suffix_x = ", log"
        
    if log_y:
        data_y = np.log(data_y)
        data_y_sm = np.log(data_y_sm)
        data_y_cn = np.log(data_y_cn)
        suffix_y = ", log"
        
    fig, ax = plt.subplots()
    x_label = ticker_x0+": "+date_st.strftime('%Y-%m-%d')+suffix_x
    y_label = ticker_x0+": "+date.strftime('%Y-%m-%d')+suffix_y
    cols = ["Country", x_label, y_label]    
    df_f = pd.concat([pd.Series(labels,name=cols[0]), pd.Series(data_x,name=cols[1]), pd.Series(data_y,name=cols[2])],axis=1)   
    
    #plot broad peers
    ax.scatter(data_x,data_y,color=(0.45, 0.45, 0.45), s=10)
    #plot small peers
    ax.scatter(data_x_sm,data_y_sm,color=mymap[0], s=10)
    #selected country
    ax.scatter(data_x_cn,data_y_cn,color=mymap[1], s=10)
    
    if y_x:
        xpoints = ypoints = ax.get_xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='r', lw=1, scalex=False, scaley=False)
        
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if labls:
        #ax.text(data_x,data_y, labels, fontsize=8,color=mymap[0]); 
        for i, txt in enumerate(labels):
            #https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_alignment.html
            ax.annotate(txt, (data_x[i], data_y[i]),ha='left', va='bottom', size=8)
    for i, txt in enumerate(labels_sm):        
        ax.annotate(txt, (data_x_sm[i], data_y_sm[i]),ha='left', va='bottom', size=8)
    
    ax.annotate(labels_cn[0], (data_x_cn[0], data_y_cn[0]),ha='left', va='bottom', size=8)
        
    plt.title(country_sel+" vs "+peers+" vs "+all_peers)
        
    plt.show()     
    st.pyplot(fig)

elif plot_type=="3. Plot: 1 indicator (x) between 2 dates - peers only":    
    norm_100 = st.checkbox('to 100',0) 
    fig, ax = plt.subplots()
    i=1
    for key in peers_s_keys:
    
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
        
        if is_x:             
            df_x = sovdb_read(ticker_x, date_st)                                    
            #st.write(df_x.empty)
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
    #st.write(df_f)
    if log_y:
        df_f = np.log(df_f)
        suffix_y = ", log"
    for col in df_f.columns:
        df_temp = df_f[col].dropna() 
        #st.write(df_temp)
        #line, = ax.plot(df_f.index, df_f[col])
        line, = ax.plot(df_temp)
        ax.annotate(col, (df_temp.index[-1], df_temp[-1]),color = line.get_color(),ha='left', va='bottom', size=8)                
    
    #fig.patch.set_facecolor((0.8, 0.8, 0.8))
    #ax.set_facecolor((0.8, 0.8, 0.8))
    
    ax.axvline(x=datetime(date.today().year-1, 12, 31), color = mymap[0],linestyle='--')
    plt.title(indic_x_eng+suffix_y+": "+peers)    
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.show()     
    st.pyplot(fig)    
                                
elif plot_type=="4. Bar: 2 indicators 1 date (end) - peers only":
    #st.write("Under construction")
    peers_tick = "PP_"+peers    
    query = "SELECT * FROM sovdb_schema.\""+peers_tick+"\""    
    cur = conn.cursor()
    cur.execute(query);            
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df_peers = pd.DataFrame(rows,columns=colnames)
    peers_m_key = df_peers.m_key
    peers_names = df_peers.country
    #st.write(peers_names)
    peers_indic_1 = []
    peers_indic_2 = []
    
    for peer in peers_m_key:
        ticker1 = peer+"_"+ticker_x0        
        query = "SELECT * FROM sovdb_schema.\""+ticker1+"\"  WHERE \"""Date\""" ='"+date.strftime('%d-%b-%Y')+"'"  
        cur = conn.cursor()
        cur.execute(query);
        rows = cur.fetchall()
        rows_x = np.array([*rows])       
        if rows_x.size!=0:
            peers_indic_1.append(rows_x[0][1])
        else:
            peers_indic_1.append(0)
 
        ticker2 = peer+"_"+ticker_y0        
        query = "SELECT * FROM sovdb_schema.\""+ticker2+"\"  WHERE \"""Date\""" ='"+date.strftime('%d-%b-%Y')+"'"  
        cur = conn.cursor()
        cur.execute(query);
        rows = cur.fetchall()
        rows_x = np.array([*rows])        
        if rows_x.size!=0:
            peers_indic_2.append(rows_x[0][1])
        else:
            peers_indic_2.append(0)
       
    df_peers_data = pd.DataFrame(
                                    {ticker_x0: peers_indic_1,                                    
                                     ticker_y0: peers_indic_2
                                    },index=peers_m_key)
    
    df_peers_data = df_peers_data.sort_values(by=[ticker_x0], ascending=False)
    df_f = df_peers_data
    barWidth = 0.25
    br1 = np.arange(len(peers_m_key))
    br2 = [x + barWidth for x in br1] 
    
    fig, ax = plt.subplots(layout='constrained')
    
    p1 = ax.bar(br1, df_peers_data[ticker_x0], color=mymap[0], label=ticker_x0,width = barWidth,) 
    ax2 = ax.twinx()
    p2 = ax2.bar(br2, df_peers_data[ticker_y0], color=mymap[1], label=ticker_y0, width = barWidth,)   
    p12 = p1+p2    
    
    #set xtick labels
    plt.xticks([r + barWidth for r in range(len(peers_m_key))], 
        df_peers_data.index)
    #rotate xtick labels
    if len(peers_m_key)>15:
        ax.set_xticklabels(df_peers_data.index,fontsize=8, rotation=90)        
        
    plt.legend(p12, [ticker_x0, ticker_y0], frameon=False)
    plt.show() 
    st.pyplot(fig)    
        
elif plot_type=="5. Bar stacked: 1 indicator (x) between 2 dates":
    fig, ax = plt.subplots()
    i=1
    for key in peers_s_keys:
    
        ticker_x = key+"_"+ticker_x0
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
   
    df_f.index = df_f.index.strftime('%Y')     
    #fig = df_f.plot.bar(stacked=True).figure
    #fig = df_f.plot.bar(stacked=True)
    ax = df_f.plot.bar(stacked=True)
    
    ax.axvline(x=datetime(date.today().year-1, 12, 31).year, color = mymap[0],linestyle='--')    
    plt.title(indic_x_eng+": "+peers)    
    #formatter = matplotlib.dates.DateFormatter('%Y')
    #ax.xaxis.set_major_formatter(formatter)
    plt.show()     
    st.pyplot(plt) 
    #st.pyplot(df_f.plot.bar(stacked=True).figure)       
    
elif plot_type=="6. Bar: 1 indicator (x) 1 date (end)":    
    cols=st.columns(4)
    with cols[0]: 
        lim_x = st.checkbox('limit x:',0) 
    with cols[1]: 
        x_shift = st.number_input('to +/-',3)
    with cols[2]: 
        y_min = st.number_input('y min',value=-1000000000)
    with cols[3]: 
        y_max = st.number_input('y max',value=1000000000)
        
    if all_peers == "All":
        all_peers_keys = m_keys
    else:
        peers_key = "PP_"+all_peers
        df = sovdb_read_gen(peers_key)        
        all_peers_keys = df.m_key 
        
    #get broad peers
    for key in all_peers_keys:
        ticker_x = key+"_"+ticker_x0
        is_x = ticker_exists(ticker_x)
        
        if is_x:             
            df_x = sovdb_read(ticker_x, date_st)            
                        
            if ~df_x.empty:            
                x_down = df_x[(df_x.index == date.strftime('%Y-%m-%d'))]                
                
                if x_down.size == 0:
                    data_x.append(0)
                else:
                    data_x.append(x_down.values[0][0])
                labels.append(key)

                
    if log_x:
        data_x = np.log(data_x)    
        suffix_x = ", log"
        
    br1 = np.arange(len(all_peers_keys))
    
    fig, ax = plt.subplots()
    x_label = ticker_x0+": "+date.strftime('%Y-%m-%d')+suffix_x
    
    cols = ["Country", ticker_x0]        
    df_f = pd.concat([pd.Series(labels,name=cols[0]), pd.Series(data_x,name=cols[1])],axis=1)           
    df_f = df_f.sort_values(by=[ticker_x0], ascending=False).reset_index(drop=True)
        
    ii = df_f[df_f.Country.isin(peers_s_keys)].index
    ij = df_f[~df_f.Country.isin(peers_s_keys)].index
    jj = df_f[df_f.Country==country_sel_key].index    
    
    p1 = ax.bar(br1, df_f[ticker_x0],color=mymap[0])    
    df_f_p = df_f.copy()
    df_f_p.loc[~df_f_p['Country'].isin(peers_s_keys) ,'Country'] = ''
    df_f_p.loc[~df_f_p['Country'].isin(peers_s_keys) ,ticker_x0] = 0
    x_lab_p = df_f_p['Country'].values
        
    df_f_c = df_f.copy()    
    df_f_c.loc[df_f_c['Country'] != country_sel_key ,'Country'] = ''
    df_f_c.loc[df_f_c['Country'] != country_sel_key ,ticker_x0] = 0
    x_lab_c = df_f_c['Country'].values
        
    x_lab = []    
    if country_sel_key in peers_s_keys.values: 
        x_lab = x_lab_p
    else:
        for i in range(len(all_peers_keys)):
            x_lab.append(x_lab_p[i]+x_lab_c[i])
    
    for i in ii:
        p1[i].set_color(mymap[1])
    p1[jj[0]].set_color('r') 
    
    plt.title(country_sel+" vs "+peers+" vs "+all_peers+": "+ticker_x0+", "+date.strftime('%Y-%m-%d'))
    plt.xticks([r for r in range(len(all_peers_keys))], x_lab)       
    ax.set_xticklabels(x_lab,fontsize=7, rotation=90) 
    ax.xaxis.set_ticks_position('none') 
        
    Countr_val = df_f[df_f.Country==country_sel_key][ticker_x0].values[0] 
    Countr_rank = jj[0]+1
    Tot_countr = len(all_peers_keys)
    Countr_str = str(round(Countr_val,1))+", rank #"+str(Countr_rank) +"/"+str(Tot_countr)
    if lim_x:
        ax.set_xticklabels(df_f['Country'].values,fontsize=10, rotation=90)
        x_left = Countr_rank-1-(x_shift+0.5)
        x_right = Countr_rank-1+(x_shift+0.5)
        if(x_left<0):
            x_left=0
        if(x_right>len(all_peers_keys)):
            x_right = len(all_peers_keys)
        
        plt.xlim((x_left, x_right))
        ymin, ymax = plt.ylim()
        plt.ylim((max(y_min, ymin), min(y_max, ymax)))
        jjj=0
        for tick in df_f[ticker_x0]:                        
            if jjj != jj[0]:
                ax.annotate(str(round(tick,1)), (jjj, tick),color = mymap[0],ha='center', va='bottom', size=8, rotation=90)  
            jjj=jjj+1
        ax.annotate(Countr_str, (jj[0], Countr_val),color = 'r',ha='center', va='bottom', size=8, rotation=90)  
    else:    
        ax.annotate(Countr_str, (jj[0], Countr_val),color = 'r',ha='center', va='bottom', size=8)  
    
    
    plt.show()     
    st.pyplot(fig)
    
cols=st.columns(3)
with cols[0]:    
    if plot_type:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
            df_f.to_excel(writer, sheet_name='Sheet1', index=True)    
        download2 = st.download_button(
            label="Excel",
            data=buffer,
            file_name=plot_type+".xlsx",
            mime='application/vnd.ms-excel'
        )
with cols[1]:    
  #  @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    if plot_type:
        csv = convert_df(df)    
        st.download_button(
            label="CSV",
            data=csv,
            file_name=plot_type+".csv",
            mime='text/csv',
        )
with cols[2]:
    fn = plot_type+".png"
    if plot_type:
        plt.savefig(fn)
        with open(fn, "rb") as img:
            btn = st.download_button(
                label="JPG",
                data=img,
                file_name=fn,
                mime="image/png"
            )     
            
st.subheader('Peers table') 
cols=st.columns(2)
with cols[0]:
    peers_t = st.checkbox('generate',0)
with cols[1]:
    peers_d = st.date_input("as of: ", pd.to_datetime('2022-12-31'))

if peers_t:
    peers_sm_key = "PP_"+peers
    df = sovdb_read_gen(peers_sm_key)
    peers_s_keys = df.m_key  
    df_p = pd.DataFrame(peers_s_keys.values)
    
    indics = ["GDP, bln USD","Popul, mln","GDP per Capita, 000 USD","GDP growth","- 10Y avg","CPI, avg","- 10Y av",\
              "GG rev","GG exp","- interest","GG bal","- 5Y avg","- pbal","GG debt","GG debt / REV","Int / REV",\
              "CA", "- 5Y avg.","Reserves", "Ext Debt", "Reserves, USD","Ext Debt, USD","Res / ExtD, %",\
              "Banks' assets","Credit growth, %","Tier1/RWA","NPL/Gross loans","ROA, %","ROE, %"]
                        
    p_tick1 = 'NGDPD_Y_WEO'    
    tick1_data = []
    p_tick2 = 'LP_Y_WEO'
    tick2_data = []
    p_tick3 = 'NGDPDPC_Y_WEO'
    tick3_data = []
    p_tick4 = 'NGDP_RPCH_Y_WEO'
    tick4_data = []
    tick4a_data = []
    p_tick5 = 'PCPIPCH_Y_WEO'
    tick5_data = []
    tick5a_data = []
    
    p_tick6 = 'GGR_NGDP_Y_WEO'
    tick6_data = []
    p_tick7 = 'GGX_NGDP_Y_WEO'
    tick7_data = []
    p_tick8 = 'GGXCNL_NGDP_Y_WEO'
    tick8_data = []
    tick8a_data = []
    p_tick9 = 'GGXONLB_NGDP_Y_WEO'
    tick9_data = []    
    p_tick10 = 'GGXWDG_NGDP_Y_WEO'
    tick10_data = []
    tick10a_data = []
    
    tick11_data = []
    tick11a_data = []

    p_tick12 = 'BCA_NGDPD_Y_WEO'
    tick12_data = []
    tick12a_data = []
    
    p_tick13 = 'DDNIIPRESGDP_Y_CUST' #Reserves, %GDP
    tick13_data = []
    tick13a_data = []    
    
    p_tick14 = 'DDEXTDUSD_Y_WB' #External Debt, USD
    tick14_data = []
    tick14a_data = []
    tick14b_data = []
    
    #MONEY
    #banks assets    
    p_tick15 = 'DDASSETSTOTBANKSGDP_Y_CUST'
    tick15_data = []
    
    #credit to private
    p_tick16 = 'DDCREDTOPRIV_Y_CUST'
    tick16_data = []
    
    #Tier1/RWA
    p_tick17 = 'DDT1RWA_Y_CUST'
    tick17_data = []
    
    #NPLs/Gross loans
    p_tick18 = 'DDNPLGL_Y_CUST'
    tick18_data = []
    
    #ROA
    p_tick19 = 'DDROA_Y_CUST'
    tick19_data = []
    
    #ROE
    p_tick20 = 'DDROE_Y_CUST'
    tick20_data = []
    
    years_shift = 10
    peers_d_p = datetime(peers_d.year-years_shift, peers_d.month, peers_d.day)    
    
    years_shift2 = 5
    peers_d_p2 = datetime(peers_d.year-years_shift2, peers_d.month, peers_d.day)    
    
    empty_strs = ["" for x in range(peers_s_keys.shape[0])]
    
    Rating_str = []
    
    for peer in peers_s_keys:        
        #Ratings
        
        if table_exists(peer+"_RATINGS"):        
            df_ratings = sovdb_read_gen(peer+"_RATINGS")    
            #st.write(df_ratings)
            Moodys_r = df_ratings.Moodys_r.values[-1]            
            SNP_r = df_ratings.SNP_r.values[-1]            
            Fitch_r = df_ratings.Fitch_r.values[-1]            
            Rating_str.append(Moodys_r+"/"+SNP_r+"/"+Fitch_r)            
        else:
            Rating_str.append("-/-/-")
        
        #GDP USD
        temp1 = sovdb_read_date(peer+"_"+p_tick1, peers_d)
        tick1_data.append(round(temp1,1))
        
        #Population
        temp = sovdb_read_date(peer+"_"+p_tick2, peers_d)
        tick2_data.append(round(temp,1))
        
        #GDP per Capita USD
        temp = sovdb_read_date(peer+"_"+p_tick3, peers_d)
        tick3_data.append(round(temp/1000,1))
        
        temp = sovdb_read_date(peer+"_"+p_tick4, peers_d)
        tick4_data.append(round(temp,1))
        
        temp = sovdb_read(peer+"_"+p_tick4, peers_d_p)
        tick4a_data.append(round(temp.values[1:1+years_shift].mean(),1))
        
        temp = sovdb_read_date(peer+"_"+p_tick5, peers_d)
        tick5_data.append(round(temp,1))
        
        temp = sovdb_read(peer+"_"+p_tick5, peers_d_p)
        tick5a_data.append(round(temp.values[1:1+years_shift].mean(),1))
        
        #fiscal
        #revenue
        temp6 = sovdb_read_date(peer+"_"+p_tick6, peers_d)
        tick6_data.append(round(temp6,1))
        
        #expenditures
        temp = sovdb_read_date(peer+"_"+p_tick7, peers_d)
        tick7_data.append(round(temp,1))
        
        #balance
        temp8 = sovdb_read_date(peer+"_"+p_tick8, peers_d)
        tick8_data.append(round(temp8,1))
        
        temp = sovdb_read(peer+"_"+p_tick8, peers_d_p2)
        tick8a_data.append(round(temp.values[1:1+years_shift2].mean(),1))
        
        #primary balance
        temp9 = sovdb_read_date(peer+"_"+p_tick9, peers_d)
        tick9_data.append(round(temp9,1))
        
        #debt
        temp10 = sovdb_read_date(peer+"_"+p_tick10, peers_d)
        tick10_data.append(round(temp10,1))
        
        #debt to revenues        
        if temp6:
            temp = temp10/temp6*100
        else:
            temp = 0
        tick10a_data.append(round(temp,1))
        
        #interest
        temp11 = temp9-temp8
        tick11_data.append(round(temp11,1))
        
        #int to revenues        
        if temp6:
            temp = temp11/temp6*100
        else:
            temp = 0
        tick11a_data.append(round(temp,1))
    
        #External
        #CA
        temp12 = sovdb_read_date(peer+"_"+p_tick12, peers_d)
        tick12_data.append(round(temp12,1))
        
        temp = sovdb_read(peer+"_"+p_tick12, peers_d_p2)
        tick12a_data.append(round(temp.values[1:1+years_shift2].mean(),1))
        
        #Reserves %GDP
        temp13 = sovdb_read_date(peer+"_"+p_tick13, peers_d)
        tick13_data.append(round(temp13,1))
        
        #Reserves, USD
        temp13a = temp13*temp1/100
        tick13a_data.append(round(temp13a,1))
        
        #External Debt, USD
        temp14 = sovdb_read_date(peer+"_"+p_tick14, peers_d)
        tick14_data.append(round(temp14/1000000000,1))
        
        #External Debt, %GDP
        if temp1:
            temp = (temp14/1000000000)/temp1*100
        else:
            temp = 0
        tick14a_data.append(round(temp,1))
        
        #Reserves / External Debt
        if temp14==0:
            tick14b_data.append(0)
        else:
            temp = temp13a/(temp14/1000000000)*100
            tick14b_data.append(round(temp,1))

        #Banks, assets
        temp15 = sovdb_read_date(peer+"_"+p_tick15, peers_d)
        if temp15==0:
            tick15_data.append(0)
        else:
            tick15_data.append(round(temp15,1))
        
        #!!!!!!!!
        #Credit growth to private
        temp16 = sovdb_read_date(peer+"_"+p_tick16, peers_d)
        if temp16==0:
            tick16_data.append(0)
        else:
            tick16_data.append(0)
            #tick16_data.append(round(temp16,1))
                    
        #Tier1 RWA
        temp17 = sovdb_read_date(peer+"_"+p_tick17, peers_d)
        if temp17==0:
            tick17_data.append(0)
        else:            
            tick17_data.append(round(temp17,1))
            
        #NPLs / Gross loans
        temp18 = sovdb_read_date(peer+"_"+p_tick18, peers_d)
        if temp18==0:
            tick18_data.append(0)
        else:            
            tick18_data.append(round(temp18,1))

        #!!!!!!!!
        #ROA
        temp19 = sovdb_read_date(peer+"_"+p_tick19, peers_d)
        if temp19==0:
            tick19_data.append(0)
        else:          
            tick19_data.append(0)
            #tick19_data.append(round(temp19,1))
            
       #!!!!!!!!            
        #ROE
        temp20 = sovdb_read_date(peer+"_"+p_tick20, peers_d)
        if temp20==0:
            tick20_data.append(0)
        else:            
            tick20_data.append(0)
#            tick20_data.append(round(temp20,1))
            
    df_p = pd.DataFrame({'M/S/F':Rating_str, 'MACRO': empty_strs, indics[0]: tick1_data,indics[1]: tick2_data,indics[2]: tick3_data,\
                         indics[3]: tick4_data,indics[4]: tick4a_data, indics[5]: tick5_data, indics[6]: tick5a_data,\
                         'FISCAL, %GDP UNO': empty_strs, indics[7]: tick6_data,  indics[8]: tick7_data, indics[9]: tick11_data, indics[10]: tick8_data, indics[11]: tick8a_data, indics[12]: tick9_data,\
                         indics[13]: tick10_data, indics[14]: tick10a_data, indics[15]: tick11a_data,\
                         'EXTERNAL, %GDP UNO': empty_strs, indics[16]: tick12_data, indics[17]: tick12a_data, indics[18]: tick13_data, indics[19]: tick14a_data, indics[20]: tick13a_data, indics[21]: tick14_data, indics[22]: tick14b_data,\
                         'MONEY, %GDP UNO': empty_strs, indics[23]: tick15_data, indics[24]: tick16_data, indics[25]: tick17_data, indics[26]: tick18_data, indics[27]: tick19_data, indics[28]: tick20_data}, index=peers_s_keys)
    
    df_p = df_p.sort_values(by=[indics[0]], ascending=False)#.reset_index(drop=True)
    df_f = df_p.transpose()
    #st.write(df_f)
    st.dataframe(
        df_f,
        hide_index=False,
        width=700,
        height=1200,
    )
    

buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
    df_f.to_excel(writer, sheet_name='Sheet1', index=True)    
download2 = st.download_button(
    label="Excel",
    data=buffer,
    file_name="PEERS-"+peers+".xlsx",
    mime='application/vnd.ms-excel'
)
         
peers_c = ""   
df_pc = sovdb_read_gen("PP_"+peers)
df_pc = df_pc.sort_values(by='country')
cntr = df_pc.country.to_list()
cm_keys = df_pc.m_key.to_list()
i=0
for c in cntr:
    if i==len(cntr)-1:
        peers_c = peers_c+str(i+1)+". "+ c + " ("+cm_keys[i]+")."
    else:
        peers_c = peers_c +str(i+1)+". "+ c + " ("+cm_keys[i]+"), "
    i = i+1

peers_c = peers +": "+ peers_c
st.write(peers_c)
#peers_t = df_mp.m_key
#if key in peers_t.to_list():
#    member_of.append(peer)
#st.write("Member of: "+', '.join(member_of))   