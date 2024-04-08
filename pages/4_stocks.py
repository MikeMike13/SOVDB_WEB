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


#all bonds
query = "SELECT * FROM sovdb_schema.equities"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
#st.write(df)
stocks_sel = df.ticker

stocks_sel_short = []
for stock in stocks_sel:
    x = stock.split("_")
    stocks_sel_short.append(x[1])

#st.write(stocks_sel_short)
cols=st.columns(5)
with cols[0]:
    ticker0 = st.selectbox("Choose stock: ",(stocks_sel_short), index=0)
with cols[1]:
    date = st.date_input("Date: ", pd.to_datetime('2022-01-01'))     
with cols[2]:
    field = st.selectbox("plot",("Close","Volume"), index=0)        
with cols[3]:
    src = st.text_input("Source","MOEX")
with cols[4]:
    reg = st.text_input("Regime","TQBR")
    
ticker = "STOCK_"+ticker0+"_"+src+"_"+reg

df = sovdb_read(ticker, date)
#st.write(df)

df_prices = df[field]

#get des of selected stock
query_des = "SELECT * FROM sovdb_schema.""equities"" WHERE ""ticker""='"+ticker+"'"
#st.write(query_des)
cur = conn.cursor()
cur.execute(query_des);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df_des = pd.DataFrame(rows,columns=colnames)
#st.write(df_des)
#st.write(df_plot)
#retruns
#1W
ret_1W_abs = (df_prices[-1]-df_prices[-5])
ret_1W_pct = (df_prices[-1]/df_prices[-5]-1)*100
#MTD
df_m = df_prices.resample('M').last()
ret_MTD_abs = (df_prices[-1]-df_m[-2])
ret_MTD_pct = (df_prices[-1]/df_m[-2]-1)*100
#YTD
df_y = df_prices.resample('Y').last()
ret_YTD_abs = (df_prices[-1]-df_y[-2])
ret_YTD_pct = (df_prices[-1]/df_y[-2]-1)*100


cols=st.columns(3)
with cols[0]:
    st.write("1W: "+str(round(ret_1W_pct,1))+"%, "+str(round(ret_1W_abs,1))+" RUB")
with cols[1]:
    st.write("MTD: "+str(round(ret_MTD_pct,1))+"%, "+str(round(ret_MTD_abs,1))+" RUB")
with cols[2]:
    st.write("YTD: "+str(round(ret_YTD_pct,1))+"%, "+str(round(ret_YTD_abs,1))+" RUB")
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
Lastdate = df.index[-1].strftime('%Y-%m-%d')
ax.plot(df[field], color=mymap[0], label='d',linewidth=0.8) 
ax.text(df_prices.index[-1], df_prices[-1], round(df_prices[-1],2), fontsize=8,color=mymap[0]);#
ax.plot(df_y.index[-2], df_y[-2], marker=5,color=(1,0,0)) 


#if bool_y:
#    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
#if bool_c and period_calc != 'no start value' and period_calc != 'no end value':
#if bool_c and Start_val*End_val:    
#    ax.plot(Start_date_c, Start_val, marker=5,color=(1,0,0)) 
#    ax.plot(End_date_c, End_val, marker=4,color=(1,0,0)) 
#    
#    
plt.title(df_des.rus_short[0]+", "+Lastdate) 

formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show() 
st.pyplot(fig)

###VS PEER
st.write('Choose peer')
cols=st.columns(2)
with cols[0]:
    is_stock = st.checkbox('use stock', 1) 
    df = df.rename(columns={"Close": ticker0+"_Price", "Volume": ticker0+"_Vol"})
    ticker_vs0 = st.selectbox("Stock: ",(stocks_sel_short), index=0)
with cols[1]:
    is_index = st.checkbox('use index', 0) 
    
if (ticker_vs0 == ticker0):
    st.warning('You have chosen the same stock')
else:
    ticker_vs = "STOCK_"+ticker_vs0+"_"+src+"_"+reg
    
    df_peer= sovdb_read(ticker_vs, date)
    df_peer = df_peer.rename(columns={"Close": ticker_vs0+"_Price", "Volume": ticker_vs0+"_Vol"})
    
    df_all = pd.concat([df, df_peer],axis=1, join="inner")  
    df_all[ticker0+"_Price_norm"] = 100*(df_all[ticker0+"_Price"] / df_all[ticker0+"_Price"].iloc[0])
    df_all[ticker_vs0+"_Price_norm"] = 100*(df_all[ticker_vs0+"_Price"] / df_all[ticker_vs0+"_Price"].iloc[0])
    
    df_all[ticker0+"_ret"] = df_all[ticker0+"_Price"].pct_change()
    df_all[ticker_vs0+"_ret"] = df_all[ticker_vs0+"_Price"].pct_change()
    #netflix_monthly_returns = netflix['Adj Close'].resample('M').ffill().pct_change()
    
    ret_ann0 = df_all[ticker0+"_ret"].mean()*252*100
    ret_ann1 = df_all[ticker_vs0+"_ret"].mean()*252*100
    ret_x = ret_ann0/ret_ann1;
    
    vol_ann0 = df_all[ticker0+"_ret"].std()*((252)**(0.5))*100
    vol_ann1 = df_all[ticker_vs0+"_ret"].std()*((252)**(0.5))*100
    vol_x = vol_ann0/vol_ann1;
    
    #st.write(df_all)
    
    cols=st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots()
        ax.plot(df_all[ticker0+"_Price_norm"], color=mymap[0], label=ticker0+" ann ret:"+str(round(ret_ann0,1))+"% ("+str(round(ret_x,1))+"x)",linewidth=0.8) 
        ax.plot(df_all[ticker_vs0+"_Price_norm"], color=mymap[1], label=ticker_vs0+" ann ret:"+str(round(ret_ann1,1))+"%",linewidth=0.8)   
        ax.axhline(100, color=(0.65,0.65,0.65))
        formatter = matplotlib.dates.DateFormatter('%Y')
        ax.xaxis.set_major_formatter(formatter)
        plt.title("Stocks dynamic: "+df_all.index[0].strftime('%d-%b-%Y')+"-"+df_all.index[-1].strftime('%d-%b-%Y'))
        plt.legend() 
        plt.show() 
        st.pyplot(fig)
    with cols[1]:    
        fig, ax = plt.subplots()    
        ax.hist(df_all[ticker0+"_ret"],100, color=mymap[0],alpha = 0.7,label=ticker0+" ann vol:"+str(round(vol_ann0,1))+"% ("+str(round(vol_x,1))+"x)")
        ax.hist(df_all[ticker_vs0+"_ret"],100, color=mymap[1],alpha = 0.7,label=ticker_vs0+" ann vol:"+str(round(vol_ann1,1))+"%")
        plt.title("Daily volatilites: "+df_all.index[0].strftime('%d-%b-%Y')+"-"+df_all.index[-1].strftime('%d-%b-%Y'))
        plt.legend() 
        plt.show() 
        st.pyplot(fig) 
        
    cols=st.columns(2)
    with cols[0]:   
        fig, ax = plt.subplots()
        ax.scatter(df_all[ticker0+"_ret"],df_all[ticker_vs0+"_ret"],color=mymap[0], s=10)
        ax.axvline(0, color=(0.65,0.65,0.65))
        ax.axhline(0, color=(0.65,0.65,0.65))
        ax.set_xlabel(ticker0)
        ax.set_ylabel(ticker_vs0)
        plt.show() 
        st.pyplot(fig)
        