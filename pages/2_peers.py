import streamlit as st
import pandas as pd
import psycopg2 as ps
import numpy as np
import matplotlib.pyplot as plt
#import math 

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];
    
conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

#get all countries m_key
query = "SELECT * FROM sovdb_schema.countries"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
m_keys = df.m_key
#st.write(df)
countries = df.name
#st.write(m_keys)

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
with cols[1]:
    peers = st.selectbox("Peers",("Peers1","Peers2"), index=0)

    
cols=st.columns(5)
with cols[0]:
    labls = st.checkbox('show labels',1) 
with cols[1]:
    log_x = st.checkbox('log x',0) 
with cols[2]:
    log_y = st.checkbox('log y',0) 
with cols[3]:
    y_x = st.checkbox('y=x',0) 
with cols[4]:
    show_all = st.checkbox('show all',1) 
    
plot_type = st.selectbox("Choose plot type",("","1. Scatter: 2 indicators 1 date (end)",\
                                             "2. Scatter: 1 indicator 2 dates",\
                                             "3. Plot: 1 indicator between 2 dates",\
                                             "4. Bar: 2 indicators 1 date",\
                                             "5. Bar stacked: 1 indicator between 2 dates"), index=0)   


if plot_type=="1. Scatter: 2 indicators 1 date (end)":
    data_x = []
    data_y = []
    labels = []
    suffix_x = ""
    suffix_y = ""
    for key in m_keys:
        ticker_x = key+"_"+ticker_x0
        #st.write(ticker_x)
        ticker_y = key+"_"+ticker_y0
        #check if exists
        query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_x+"'"    
       # cur = conn.cursor()
        cur.execute(query);
        rows_x = cur.fetchall()
        rows_xx = np.array([*rows_x])
        colnames = [desc[0] for desc in cur.description]
        df_x = pd.DataFrame(rows_x,columns=colnames)
        
        query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_y+"'"    
       # cur = conn.cursor()
        cur.execute(query);
        rows_y = cur.fetchall()
        rows_yy = np.array([*rows_y])
        colnames = [desc[0] for desc in cur.description]
        df_y = pd.DataFrame(rows_y,columns=colnames)
        
        if rows_xx.size !=0 and rows_yy.size != 0:        
            #st.write(ticker_x)
            #st.write(df_x.index[-1].values)
            #if df_x.end_date.values > date and df_y.end_date.values > date:
            query = "SELECT * FROM sovdb_schema.\""+ticker_x+"\""    
            cur.execute(query);
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            df_x_d = pd.DataFrame(rows,columns=colnames)
            df_x_d = pd.DataFrame(df_x_d).set_index('Date')
            df_x_d.index = pd.to_datetime(df_x_d.index)
            
            query = "SELECT * FROM sovdb_schema.\""+ticker_y+"\""    
            cur.execute(query);
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            df_y_d = pd.DataFrame(rows,columns=colnames)
            df_y_d = pd.DataFrame(df_y_d).set_index('Date')
            df_y_d.index = pd.to_datetime(df_y_d.index)     
            
            
            if df_x_d.index[-1].date() >= date and df_y_d.index[-1].date() >= date:
                #st.write(df_x.index[-1].values)
                
                #query = "SELECT * FROM sovdb_schema.\""+ticker_x+"\"  WHERE \"""Date\""" ='"+date.strftime('%d-%b-%Y')+"'"    
                #cur.execute(query);
                #rows = cur.fetchall()
                #rows = np.array([*rows])
                #st.write(key)
                #st.write(df_x_d[(df_x_d.index == date.strftime('%Y-%m-%d')) ])
                x_down = df_x_d[(df_x_d.index == date.strftime('%Y-%m-%d'))]
                y_down = df_y_d[(df_y_d.index == date.strftime('%Y-%m-%d'))]
                #st.write(type(x_down))
                if x_down.size == 0:
                    data_x.append(0)
                else:
                    data_x.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y.append(0)
                else:
                    data_y.append(y_down.values[0][0])
                labels.append(key)
        
    #st.write(data_x)    
    #st.write(data_y)    
    #st.write(labels) 
    
    fig, ax = plt.subplots()
    #Lastdate = df.index[-1].strftime('%Y-%m-%d')
    #st.write(colnames)
    if log_x:
        data_x = np.log(data_x)
        suffix_x = ", log"
    if log_y:
        data_y = np.log(data_y)  
        suffix_y = ", log"
        
    ax.scatter(data_x,data_y,color=(0.45, 0.45, 0.45), s=10)
    if y_x:
        xpoints = ypoints = ax.get_xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='r', lw=1, scalex=False, scaley=False)
        
    ax.set_xlabel(ticker_x0+suffix_x)
    ax.set_ylabel(ticker_y0+suffix_x)
    if labls:
        #ax.text(data_x,data_y, labels, fontsize=8,color=mymap[0]); 
        for i, txt in enumerate(labels):
            #https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_alignment.html
            ax.annotate(txt, (data_x[i], data_y[i]),ha='left', va='bottom', size=8)
    #st.write(type(df))
    #if bool_y:
    #    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
    #if bool_c and period_calc != 'no start value' and period_calc != 'no end value':
    #if bool_c and Start_val*End_val:    
    #    ax.plot(Start_date_c, Start_val, marker=5,color=(1,0,0)) 
    #    ax.plot(End_date_c, End_val, marker=4,color=(1,0,0)) 
    #plt.title(ticker+", "+Lastdate) 
    #plt.legend()     
    plt.show()     
    st.pyplot(fig)
elif plot_type=="2. Scatter: 1 indicator 2 dates":
    data_x = []
    data_y = []
    labels = []
    suffix_x = ""
    suffix_y = ""
    for key in m_keys:
        ticker_x = key+"_"+ticker_x0
        #st.write(ticker_x)
        #ticker_y = key+"_"+ticker_y0
        #check if exists
        query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_x+"'"    
       # cur = conn.cursor()
        cur.execute(query);
        rows_x = cur.fetchall()
        rows_xx = np.array([*rows_x])
        colnames = [desc[0] for desc in cur.description]
        df_x = pd.DataFrame(rows_x,columns=colnames)
        
        #query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_y+"'"    
       # cur = conn.cursor()
        #cur.execute(query);
        #rows_y = cur.fetchall()
        #rows_yy = np.array([*rows_y])
        #colnames = [desc[0] for desc in cur.description]
        #df_y = pd.DataFrame(rows_y,columns=colnames)
        
        if rows_xx.size !=0: # and rows_yy.size != 0:        
            #st.write(ticker_x)
            #st.write(df_x.index[-1].values)
            #if df_x.end_date.values > date and df_y.end_date.values > date:
            query = "SELECT * FROM sovdb_schema.\""+ticker_x+"\""    
            cur.execute(query);
            rows = cur.fetchall()
            colnames = [desc[0] for desc in cur.description]
            df_x_d = pd.DataFrame(rows,columns=colnames)
            df_x_d = pd.DataFrame(df_x_d).set_index('Date')
            df_x_d.index = pd.to_datetime(df_x_d.index)
            
            #query = "SELECT * FROM sovdb_schema.\""+ticker_y+"\""    
            #cur.execute(query);
            #rows = cur.fetchall()
            #colnames = [desc[0] for desc in cur.description]
            #df_y_d = pd.DataFrame(rows,columns=colnames)
            #df_y_d = pd.DataFrame(df_y_d).set_index('Date')
            #df_y_d.index = pd.to_datetime(df_y_d.index)     
            
            
            if df_x_d.index[-1].date() >= date:# and df_y_d.index[-1].date() >= date:
                #st.write(df_x.index[-1].values)
                
                #query = "SELECT * FROM sovdb_schema.\""+ticker_x+"\"  WHERE \"""Date\""" ='"+date.strftime('%d-%b-%Y')+"'"    
                #cur.execute(query);
                #rows = cur.fetchall()
                #rows = np.array([*rows])
                #st.write(key)
                #st.write(df_x_d[(df_x_d.index == date.strftime('%Y-%m-%d')) ])
                x_down = df_x_d[(df_x_d.index == date_st.strftime('%Y-%m-%d'))]
                y_down = df_x_d[(df_x_d.index == date.strftime('%Y-%m-%d'))]
                #st.write(type(x_down))
                if x_down.size == 0:
                    data_x.append(0)
                else:
                    data_x.append(x_down.values[0][0])
                    
                if y_down.size == 0:
                    data_y.append(0)
                else:
                    data_y.append(y_down.values[0][0])
                labels.append(key)
        
    #st.write(data_x)    
    #st.write(data_y)    
    #st.write(labels) 
    if log_x:
        data_x = np.log(data_x)
        suffix_x = ", log"
        
    if log_y:
        data_y = np.log(data_y)
        suffix_y = ", log"
        
    #st.write(log_x)
    fig, ax = plt.subplots()
    #Lastdate = df.index[-1].strftime('%Y-%m-%d')
    #st.write(colnames)
    ax.scatter(data_x,data_y,color=(0.45, 0.45, 0.45), s=10)
    if y_x:
        xpoints = ypoints = ax.get_xlim()
        ax.plot(xpoints, ypoints, linestyle='--', color='r', lw=1, scalex=False, scaley=False)
        
    ax.set_xlabel(ticker_x0+": "+date_st.strftime('%Y-%m-%d')+suffix_x)
    ax.set_ylabel(ticker_x0+": "+date.strftime('%Y-%m-%d')+suffix_y)
    if labls:
        #ax.text(data_x,data_y, labels, fontsize=8,color=mymap[0]); 
        for i, txt in enumerate(labels):
            #https://matplotlib.org/stable/gallery/text_labels_and_annotations/text_alignment.html
            ax.annotate(txt, (data_x[i], data_y[i]),ha='left', va='bottom', size=8)
    #st.write(type(df))
    #if bool_y:
    #    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
    #if bool_c and period_calc != 'no start value' and period_calc != 'no end value':
    #if bool_c and Start_val*End_val:    
    #    ax.plot(Start_date_c, Start_val, marker=5,color=(1,0,0)) 
    #    ax.plot(End_date_c, End_val, marker=4,color=(1,0,0)) 
    #plt.title(ticker+", "+Lastdate) 
    #plt.legend()     
    plt.show()     
    st.pyplot(fig)
elif plot_type=="3. Plot: 1 indicator between 2 dates":
    st.write("Under construction")
elif plot_type=="4. Bar: 2 indicators 1 date":
    st.write("Under construction")
elif plot_type=="5. Bar stacked: 1 indicator between 2 dates":
    st.write("Under construction")    