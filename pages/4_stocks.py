import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import matplotlib 
import matplotlib.pyplot as plt
#from datetime import date, datetime

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

#all bonds
query = "SELECT * FROM sovdb_schema.equities"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
#st.write(df)
bond_sel = df.ticker

cols=st.columns(2)
with cols[0]:
    ticker = st.selectbox("Choose bond: ",(bond_sel), index=0)
with cols[1]:
    field = st.selectbox("plot",("Close","Volume"), index=0)        

#get data for selected bond
query = "SELECT * FROM sovdb_schema.\""+ticker+"\""
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
df = pd.DataFrame(df).set_index('Date')
df = df.sort_index()
df_plot = df[field]#count_sel = df.id
#st.write(df)

#get des of selected bond
query_des = "SELECT * FROM sovdb_schema.""bonds"" WHERE ""id""='"+ticker+"'"
#st.write(query_des)
cur = conn.cursor()
cur.execute(query_des);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df_des = pd.DataFrame(rows,columns=colnames)

#cols=st.columns(4)
#with cols[0]:
#    name = st.write(df_des.rus_long.values[0])
#with cols[1]:
#    cpn = st.write("cpn: "+str(round(df_des.cpn_rate.values[0],2))+"%")
#with cols[2]:
#    mat = st.write("maturity: "+str(df_des.maturity_date.values[0]))
#    #mat = datetime.strptime(df_des.maturity_date.values[0], '%y-%m-%d')
#    #st.write(type(mat))
#with cols[3]:    
#    years = (df_des.maturity_date.values[0] - date.today()).days/365.25
#    years_mat = st.write("years to mat: "+str(round(years,2))+"Y")
#plot selected bond    

fig, ax = plt.subplots()
Lastdate = df_plot.index[-1].strftime('%Y-%m-%d')
##st.write(colnames)
ax.plot(df_plot, color=mymap[0], label='d',linewidth=0.8) 
##st.write(type(df))
ax.text(df_plot.index[-1], df_plot[-1], round(df_plot[-1],2), fontsize=8,color=mymap[0]);#

#if bool_y:
#    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
#if bool_c and period_calc != 'no start value' and period_calc != 'no end value':
#if bool_c and Start_val*End_val:    
#    ax.plot(Start_date_c, Start_val, marker=5,color=(1,0,0)) 
#    ax.plot(End_date_c, End_val, marker=4,color=(1,0,0)) 
#    
#    
plt.title(ticker+", "+Lastdate) 
#plt.legend() 
#
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show() 
st.pyplot(fig)
