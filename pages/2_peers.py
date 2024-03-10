import streamlit as st
import pandas as pd
import psycopg2 as ps
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import io
import matplotlib

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

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

    
cols=st.columns(5)
with cols[0]:
    labls = st.checkbox('all peers labels',0) 
with cols[1]:
    log_x = st.checkbox('log x',0) 
with cols[2]:
    log_y = st.checkbox('log y',0) 
with cols[3]:
    y_x = st.checkbox('y=x',0) 
with cols[4]:
    all_peers = st.selectbox("All peers",("WEO","EM","DM","All"), index=0)
    
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
        data_x = np.log(data_x)
        data_x_sm = np.log(data_x_sm)
        data_x_cn = np.log(data_x_cn)
        suffix_x = ", log"
    if log_y:
        data_y = np.log(data_y)  
        data_y_sm = np.log(data_y_sm)
        data_y_cn = np.log(data_y_cn)
        suffix_y = ", log"

    ticker_x0        
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