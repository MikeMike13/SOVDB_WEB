import streamlit as st
import psycopg2 as ps
import pandas as pd
import numpy as np
#import math
import matplotlib 
import matplotlib.pyplot as plt
#from datetime import datetime
from scipy.interpolate import splrep, BSpline
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="centered")

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)
##DURATION
##https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/TQCB/securities/RU000A1026B3.xml?from=2024-04-01&till=2024-04-11

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
    df = df.sort_index()
    return df
   
def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

def sovdb_read_des(tbl, ticker):
    #get des of selected bond
    query_des = "SELECT * FROM sovdb_schema."+tbl+" WHERE ""id""='"+ticker+"'"
    #st.write(query_des)
    cur = conn.cursor()
    cur.execute(query_des);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df_des = pd.DataFrame(rows,columns=colnames)
    return df_des

all_bonds = sovdb_read_gen('bonds')
#st.write(all_bonds)
all_bonds0 = all_bonds[all_bonds['is_matured']==False]

cols=st.columns(4)
with cols[0]:
    #st.write(all_bonds[all_bonds['is_matured']==False]['rus_short'])
    country = st.selectbox("Country: ",(['Russia','United States','Kazakhstan']), index=0)
    all_bonds = all_bonds0[all_bonds0['Country']==country]
    #ticker      = temp['id'].array[0]
    #%ticker_isin = temp['isin'].array[0]    
with cols[1]:
    #st.write(all_bonds[all_bonds['is_matured']==False]['rus_short'])
    ticker0 = st.selectbox("Choose bond: ",(all_bonds['rus_short'].sort_values()), index=0)
    temp = all_bonds[all_bonds['rus_short']==ticker0]
    ticker      = temp['id'].array[0]
    ticker_isin = temp['isin'].array[0]    
with cols[2]:
    date = st.date_input("Date: ", pd.to_datetime('2022-01-01'))  
with cols[3]:
    field0 = st.selectbox("plot",("Yield_Close","Price_Close"), index=0)        

#get data for selected bond
df0 = sovdb_read(ticker,date)
#st.write(df0)
df = df0[['Price_Close','Yield_Close','Volume']]
df = df.rename(columns={"Price_Close": ticker+"_Price_Close", "Yield_Close": ticker+"_Yield_Close", "Volume": ticker+"_Vol"})
field = ticker+"_"+field0
field_vol = ticker+"_Vol"

df_des = sovdb_read_des("bonds",ticker)
#st.write(df_des)

cols=st.columns(5)
with cols[0]:
    name = st.write(df_des.rus_long.values[0])
with cols[1]:
    #st.write(type(df_des.cpn_rate.values[0]))
    if df_des.cpn_rate.values[0] is None:
        cpn = st.write("cpn: -")
    else:
        cpn = st.write("cpn: "+str(round(df_des.cpn_rate.values[0],2))+"%")
with cols[2]:
    mat = st.write("maturity: "+str(df_des.maturity_date.values[0]))
    #mat = datetime.strptime(df_des.maturity_date.values[0], '%y-%m-%d')
    #st.write(type(mat))
with cols[3]:    
    if df_des.maturity_date.values[0] is None:
        st.write("years to mat: -")
    else:
        years = (df_des.maturity_date.values[0] - date.today()).days/365.25
        years_mat = st.write("years to mat: "+str(round(years,2))+"Y")
with cols[4]:    
    ticker_dur = df0['Duration'].values[-1]/365.25
    st.write("dur: "+str(round(ticker_dur,1)))
    
#plot selected bond    
fig, ax = plt.subplots()
Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
ax.plot(df[field], color=mymap[0], label='d',linewidth=0.8) 
ax.text(df[field].index[-1], df[field][-1], round(df[field][-1],2), fontsize=8,color=mymap[0]);#
ax2 = ax.twinx()
ax2.bar(df.index,df[field_vol]/1000, color=mymap[1]) 

plt.title(ticker0+", "+Lastdate) 
formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)
plt.show() 
st.pyplot(fig)

cols=st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    ax.scatter(df[ticker+"_Yield_Close"],df[ticker+"_Price_Close"],color=mymap[0], s=10,alpha=0.5)
    ax.scatter(df[ticker+"_Yield_Close"][-1],df[ticker+"_Price_Close"][-1],color=(1,0,0), s=10)
    ax.set_xlabel('yield')
    ax.set_ylabel('price')
    plt.show() 
    st.pyplot(fig)    
with cols[1]:
    last_n_days = 20
    model = LinearRegression()
    x = np.array(df[ticker+"_Yield_Close"].tail(last_n_days).values.tolist()).reshape((-1, 1))
    y = np.array(df[ticker+"_Price_Close"].tail(last_n_days).values.tolist())
    model.fit(x, y)
    b = model.intercept_
    k = model.coef_
    
    x_new = np.arange(np.min(x),  np.max(x), 0.01)
    y_new = k*x_new+b
    
    
    fig, ax = plt.subplots()
    ax.scatter(df[ticker+"_Yield_Close"].tail(last_n_days),df[ticker+"_Price_Close"].tail(last_n_days),color=mymap[0], s=10,alpha=0.5)
    ax.scatter(df[ticker+"_Yield_Close"][-1],df[ticker+"_Price_Close"][-1],color=(1,0,0), s=10)
    ax.text(df[ticker+"_Yield_Close"][-1],df[ticker+"_Price_Close"][-1],str(round(k[0],2)))
    ax.plot(x_new,y_new)
    ax.set_xlabel('yield')
    ax.set_ylabel('price')
    plt.show() 
    st.pyplot(fig) 

cols=st.columns(2)
with cols[0]:
    #st.write(all_bonds[all_bonds['is_matured']==False]['rus_short'])
    country_peer = st.selectbox("Country : ",(['Russia','United States','Kazakhstan']), index=0)
    all_bonds = all_bonds0[all_bonds0['Country']==country_peer]
    #ticker      = temp['id'].array[0]
    #%ticker_isin = temp['isin'].array[0]    
with cols[1]:
    #st.write(all_bonds[all_bonds['is_matured']==False]['rus_short'])
    #ticker0 = st.selectbox("Choose bond: ",(all_bonds['rus_short'].sort_values()), index=0)
    #temp = all_bonds[all_bonds['rus_short']==ticker0]
    #ticker      = temp['id'].array[0]
    #ticker_isin = temp['isin'].array[0]    
    
    ticker0_vs = st.selectbox("choose peer",(all_bonds['rus_short'].sort_values(ascending=True)), index=1)  
    temp = all_bonds[all_bonds['rus_short']==ticker0_vs]
    ticker_peer      = temp['id'].array[0]
    ticker_isin_peer = temp['isin'].array[0]

df_peer = sovdb_read(ticker_peer,date)
df_peer = df_peer[['Price_Close','Yield_Close','Volume']]
df_peer = df_peer.rename(columns={"Price_Close": ticker_peer+"_Price_Close", "Yield_Close": ticker_peer+"_Yield_Close", "Volume": ticker_peer+"_Vol"})
df_all = pd.concat([df, df_peer],axis=1, join="inner")  
df_all['Spread'] = (df_all[ticker+"_Yield_Close"] - df_all[ticker_peer+"_Yield_Close"])*100
field_peer = ticker_peer+"_"+field0
field_vol_peer = ticker_peer+"_Vol"

df_all[ticker+"_Price_Close_norm"] = 100*(df_all[ticker+"_Price_Close"] / df_all[ticker+"_Price_Close"].iloc[0])
df_all[ticker_peer+"_Price_Close_norm"] = 100*(df_all[ticker_peer+"_Price_Close"] / df_all[ticker_peer+"_Price_Close"].iloc[0])

cols=st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.plot(df_all[field], color=mymap[0], label=ticker0,linewidth=0.8) 
    ax.text(df_all[field].index[-1], df_all[field][-1], round(df_all[field][-1],2), fontsize=8,color=mymap[0]);#
    
    ax.plot(df_all[field_peer], color=mymap[1], label=ticker0_vs,linewidth=0.8) 
    ax.text(df_all[field_peer].index[-1], df_all[field_peer][-1], round(df_all[field_peer][-1],2), fontsize=8,color=mymap[1]);#

    plt.title("Yields: "+ticker0+" vs "+ticker0_vs+", "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)
with cols[1]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.fill_between(df_all.index,df_all['Spread'], color=mymap[0], label='Spread',linewidth=0.8) 
    ax.text(df_all['Spread'].index[-1], df_all['Spread'][-1], round(df_all['Spread'][-1],0), fontsize=8,color=mymap[0]);#
        
    plt.title("Spread: "+ticker0+" vs "+ticker0_vs+", "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)

cols=st.columns(2)
with cols[0]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.plot(df_all[ticker+"_Price_Close"], color=mymap[0], label=ticker0,linewidth=0.8) 
    ax.text(df_all[ticker+"_Price_Close"].index[-1], df_all[ticker+"_Price_Close"][-1], round(df_all[ticker+"_Price_Close"][-1],2), fontsize=8,color=mymap[0]);#
    
    ax.plot(df_all[ticker_peer+"_Price_Close"], color=mymap[1], label=ticker0_vs,linewidth=0.8) 
    ax.text(df_all[ticker_peer+"_Price_Close"].index[-1], df_all[ticker_peer+"_Price_Close"][-1], round(df_all[ticker_peer+"_Price_Close"][-1],2), fontsize=8,color=mymap[1]);#
    
    plt.title("Prices: "+ticker0+" vs "+ticker0_vs+", "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)
    
with cols[1]:
    fig, ax = plt.subplots()
    Lastdate = df[field].index[-1].strftime('%Y-%m-%d')
    ax.plot(df_all[ticker+"_Price_Close_norm"], color=mymap[0], label=ticker0,linewidth=0.8) 
    ax.text(df_all[ticker+"_Price_Close_norm"].index[-1], df_all[ticker+"_Price_Close_norm"][-1], round(df_all[ticker+"_Price_Close_norm"][-1],2), fontsize=8,color=mymap[0]);#
    
    ax.plot(df_all[ticker_peer+"_Price_Close_norm"], color=mymap[1], label=ticker0_vs,linewidth=0.8) 
    ax.text(df_all[ticker_peer+"_Price_Close_norm"].index[-1], df_all[ticker_peer+"_Price_Close_norm"][-1], round(df_all[ticker_peer+"_Price_Close_norm"][-1],2), fontsize=8,color=mymap[1]);#
    ax.axhline(100, color=(0.45,0.45,0.45))
    plt.title("Prices norm: "+ticker0+" vs "+ticker0_vs+", "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)
    
cols=st.columns(2)
with cols[0]:
    ticker_vol_mean = df_all[field_vol].mean()/1000
    ticker_vol_peer_mean = df_all[field_vol_peer].mean()/1000
    ticker_vol_x = ticker_vol_mean/ticker_vol_peer_mean
    
    ticker_vol_mean_1M = df_all[field_vol].tail(20).mean()/1000
    ticker_vol_peer_mean_1M = df_all[field_vol_peer].tail(20).mean()/1000
    ticker_vol_x_1M = ticker_vol_mean_1M/ticker_vol_peer_mean_1M

    
    fig, ax = plt.subplots()
    Lastdate = df_all[field].index[-1].strftime('%Y-%m-%d')
    ax.bar(df_all.index,df_all[field_vol]/1000, color=mymap[1],label=ticker0+": "+str(round(ticker_vol_mean,1))+" (1M "+str(round(ticker_vol_mean_1M,1))+")") 
    ax.bar(df_all.index,df_all[field_vol_peer]/1000, color=mymap[2], label=ticker0_vs+": "+str(round(ticker_vol_peer_mean,1))+" (1M "+str(round(ticker_vol_peer_mean_1M,1))+")") 
    ax.axhline(ticker_vol_mean, color=mymap[1])
    ax.axhline(ticker_vol_peer_mean, color=mymap[2])

    plt.title(ticker0+" vs "+ticker0_vs+", ("+str(round(ticker_vol_x,3))+"x/"+str(round(ticker_vol_x_1M,3))+"x) "+Lastdate) 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.legend()
    plt.show() 
    st.pyplot(fig)
    
# =============================================================================
# st.write('All bonds by duration')
# fields_to_show = ['isin','rus_short','maturity_date','years_to_maturity','duration']
# years_to_maturity = []
# duration = []
# 
# for inx, bond in all_bonds.iterrows():
#     years_to_maturity.append((bond['maturity_date'] - date.today()).days/365.25)
#     
#     df01 = sovdb_read(bond['id'],date)   
#     
#     if df01['Duration'].empty:
#         duration.append(0)
#     else:
#         duration.append(df01['Duration'].values[-1]/365.25)
#     
# all_bonds['years_to_maturity'] = years_to_maturity
# all_bonds['duration'] = duration
# 
# st.write(all_bonds[fields_to_show].sort_values(by=['duration']))
# =============================================================================

##CURVE
Curve0   = ["SU26234RMFS3","SU26229RMFS3","SU26236RMFS8","SU26228RMFS5","SU26221RMFS0","SU26230RMFS1","SU26238RMFS4"];

Curve = ["BOND_" + sub for sub in Curve0] 
Curve = [sub + "_MOEX_TQOB" for sub in Curve]

curve_df = pd.DataFrame()
curve_mat = []
for bond in Curve:
    temp = sovdb_read(bond, date)
    curve_df_temp = temp['Yield_Close']
    curve_df_temp = curve_df_temp.rename(bond+"_Yield_Close").to_frame()
    curve_df = pd.concat([curve_df, curve_df_temp],axis=1,)
    
    temp = sovdb_read_des("bonds",bond)
    #st.write(temp)
    curve_mat.append(temp.maturity_date.values[0])

cols=st.columns(4)
with cols[0]:
    curve_date_st = st.date_input("Start Date: ", curve_df.index[-20])  
with cols[1]:
    curve_date_end = st.date_input("End Date: ", curve_df.index[-1]) 
with cols[2]:
    s_st = st.number_input('Smooth st',value=0.03,format="%.4f",step=0.001)
with cols[3]:
    s_end = st.number_input('Smooth end',value=0.08,format="%.4f",step=0.001)

dt_st = []
dt_end = []    
for mat in curve_mat:
    dt_st.append((mat - curve_date_st).days/365.25)
    dt_end.append((mat - curve_date_end).days/365.25)
    
st_values = curve_df[curve_df.index==curve_date_st.strftime('%Y-%m-%d')]
end_values = curve_df[curve_df.index==curve_date_end.strftime('%Y-%m-%d')]

st_spline = splrep(dt_st, st_values.values.tolist()[0], s=s_st)
st_xnew = np.arange(np.ceil(dt_st[0])-1, np.ceil(dt_st[-1])+1, 0.5)
st_value_interploated = BSpline(*st_spline)(st_xnew)

end_spline = splrep(dt_end, end_values.values.tolist()[0], s=s_end)
end_xnew = np.arange(np.ceil(dt_end[0])-1, np.ceil(dt_end[-1])+1, 0.5)
end_value_interploated = BSpline(*end_spline)(end_xnew)

#selected bond
#st.write(st_values_b)

dt_st_b = (df_des.maturity_date.values[0] - curve_date_st).days/365.25
dt_end_b = (df_des.maturity_date.values[0] - curve_date_end).days/365.25

st_values_b = df[df.index==curve_date_st.strftime('%Y-%m-%d')]
st_values_b = st_values_b[ticker+"_Yield_Close"].values[0]
st_spread_b = (st_values_b - BSpline(*st_spline)(dt_st_b))*100


end_values_b = df[df.index==curve_date_end.strftime('%Y-%m-%d')]
end_values_b = end_values_b[ticker+"_Yield_Close"].values[0]
end_spread_b = (end_values_b - BSpline(*end_spline)(dt_end_b)  )*100

fig, ax = plt.subplots()
ax.scatter(dt_st,st_values,color=mymap[0], label=curve_date_st.strftime('%Y-%m-%d'),s=10)
ax.scatter(dt_st_b,st_values_b,color=mymap[0], label=ticker0+": "+curve_date_st.strftime('%Y-%m-%d'),s=10,marker='^')
ax.plot(st_xnew, st_value_interploated, '-',color=mymap[0])
ax.text(dt_st_b,st_values_b, str(round(st_spread_b,1)),color=mymap[0])

ax.scatter(dt_end,end_values,color=mymap[1], label=curve_date_end.strftime('%Y-%m-%d'), s=10)
ax.scatter(dt_end_b,end_values_b,color=mymap[1], label=ticker0+": "+curve_date_st.strftime('%Y-%m-%d'),s=10,marker='^')
ax.plot(end_xnew, end_value_interploated, '-',color=mymap[1])
ax.text(dt_end_b,end_values_b, str(round(end_spread_b,1)),color=mymap[1])

ax.set_xlabel('years to maturity')
ax.set_ylabel('yield')
plt.legend()
plt.title('Yield curve')
plt.show() 
st.pyplot(fig) 

list1 = end_values.values.tolist()[0]
list2 = st_values.values.tolist()[0]
result = [(a - b)*100 for a, b in zip(list1, list2)]

fig, ax = plt.subplots()
ax.bar(dt_end,result)


ax.set_xlabel('years to maturity')
ax.set_ylabel('spread, bp')
plt.legend()
plt.title('Yield curve shift')
plt.show() 
st.pyplot(fig) 


curve_bonds = all_bonds[all_bonds['id'].isin(Curve)].sort_values(by=['maturity_date'])
ddd = (curve_bonds.maturity_date.values - date.today()).tolist()
curve_bonds['years_to_mat'] =  [round(x.days/365.25,2) for x in ddd]
st.write(curve_bonds[['rus_long','issue_date','maturity_date','years_to_mat']])    