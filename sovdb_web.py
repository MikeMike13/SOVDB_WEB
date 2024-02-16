import streamlit as st
import matplotlib 
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2 as ps
import datetime
from datetime import date
from io import BytesIO
import io

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    format1 = workbook.add_format({'num_format': '0.00'}) 
    worksheet.set_column('A:A', None, format1)  
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def decum(df,FREQ):
    de_cum = []
    dates = []
    for i in range(0, len(df)):
        dates.append(df.index[i])
        match FREQ: 
            case "M":
                if df.index[i].month == 1:
                    #de_cum.append(df.iloc[i].Value)
                    de_cum.append(df[i])                  
                else:
                    #de_cum.append(df.iloc[i].Value-df.iloc[i-1].Value)  
                    de_cum.append(df[i]-df[i-1])  
    
    df = pd.DataFrame({'Date':dates, 'Value':de_cum})
    df = pd.DataFrame(df).set_index('Date')
    df = df.squeeze()
    #st.write(df)
    return df

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

st.write("Available data: [link](https://docs.google.com/spreadsheets/d/15xyVYbzi04rBfxX4c9wfemcV-BD0pjvb6oCO-2KPGnM/edit#gid=0)")

cols=st.columns(2)
with cols[0]:
    Start_date = st.date_input("Start: ", datetime.date(2022, 1, 1))
with cols[1]:    
    End_date = st.date_input("End: ", date.today())
    
cols=st.columns(2)
with cols[0]:    
    ticker = st.text_input('Ticker', 'FX_RUBUSD_CBR')
with cols[1]:
    field = st.selectbox("Field",("Value","Yield_Close", "Price_Close","Close","Volume"), index=0)
    
cols=st.columns(7)
with cols[0]:
    freq = st.selectbox("FREQ",("none","D","M", "Q", "Y"), index=None)
with cols[1]:
    method = st.selectbox("METH",("none","EOP", "AVG","SUM","DECUM"), index=None)
with cols[2]:
    trans = st.selectbox("Transform",("none","mom", "qoq","yoy"), index=None)
with cols[3]:    
    func = st.selectbox("Function",("none","sum","std", "cmlt","avg"), index=None)
with cols[4]:    
    window_num = st.number_input('Window')
with cols[5]:    
    bool_y = st.checkbox('Y')
with cols[6]:    
    y_level = st.number_input('level')    

#BOND_SU26219RMFS4_MOEX_TQOB
#STOCK_AFLT_MOEX_TQBR

query = "SELECT * FROM sovdb_schema.\""+ticker+"\"";
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
#st.write(colnames)
df = pd.DataFrame(rows,columns=colnames)
df = pd.DataFrame(df).set_index('Date')
df = df.sort_index()
df = df[field]
df.index = pd.to_datetime(df.index)

df = df[(df.index >= Start_date.strftime('%Y-%m-%d')) & (df.index <= End_date.strftime('%Y-%m-%d'))]

#st.write(df)

if freq=="M":
    match method:
        case "EOP":
            df = df.resample(freq).last()
        case "AVG":
            df = df.resample(freq).mean()
        case "SUM":
            df = df.resample(freq).sum()
        case "DECUM":
            df = decum(df,freq)
elif freq=="Q":
    match method:
        case "EOP":
            df = df.resample('Q').last()
        case "AVG":
            df = df.resample('Q').mean()
        case "SUM":
            df = df.resample('Q').sum()
elif freq=="Y":
    match method:
        case "EOP":
            df = df.resample('Y').last()
        case "AVG":
            df = df.resample('Y').mean()
        case "SUM":
            df = df.resample('Y').sum()           
match freq:
    case "D":
        match trans:
            case "mom":
                df = df.pct_change(periods=20) * 100
            case "qoq":
                df = df.pct_change(periods=3*20) * 100
            case "yoy":
                df = df.pct_change(periods=252) * 100 
    case "M":
        match trans:
            case "mom":
                df = df.pct_change(periods=1) * 100
            case "qoq":
                df = df.pct_change(periods=4) * 100
            case "yoy":
                df = df.pct_change(periods=12) * 100        
    case "Q":
        match trans:            
            case "qoq":
                df = df.pct_change(periods=1) * 100
            case "yoy":
                df = df.pct_change(periods=4) * 100        
    case "Y":
        match trans:                        
            case "yoy":
                df = df.pct_change(periods=1) * 100        
            
match func:
    case "avg":
        df = df.rolling(window=int(window_num)).mean()
    case "sum":
         df = df.rolling(int(window_num)).sum()
    case "std":
        df = df.rolling(int(window_num)).std().shift()
    case "cmlt":
        df = (1 + df/100).cumprod() - 1

                
fig, ax = plt.subplots()
Lastdate = df.index[-1].strftime('%Y-%m-%d')
#st.write(colnames)
ax.plot(df, color=mymap[0], label='d',linewidth=0.8) 
#st.write(type(df))
ax.text(df.index[-1], df[-1], round(df[-1],2), fontsize=8,color=mymap[0]);

if bool_y:
    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)

plt.title(ticker+", "+Lastdate) 
plt.legend() 

formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)

plt.show() 

st.pyplot(fig)


buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
    df.to_excel(writer, sheet_name='Sheet1', index=True)    
download2 = st.download_button(
    label="Excel",
    data=buffer,
    file_name=ticker+".xlsx",
    mime='application/vnd.ms-excel'
)

query = "SELECT * FROM sovdb_schema.countries"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
count_sel = df.name

#query2 = "SELECT * FROM sovdb_schema.macro_indicators";
#st.write(query2)
#cur = conn.cursor()
#cur.execute(query2);
#rows = cur.fetchall()
#colnames = [desc[0] for desc in cur.description]
#df = pd.DataFrame(rows,columns=colnames)
#groups = df.group.unique()

#budget - DROP
groups = ['real','external','fiscal','popul','markets','eco','covid','finance','institute','budget','all']

tot_str = "("
for i in range(0,len(groups)-1):
    tot_str = tot_str+"'"+groups[i]+"', "
tot_str = tot_str[:-2]
tot_str = tot_str+")"

cols=st.columns(2)
with cols[0]:
    countr = st.selectbox("Country",(count_sel), index=203)
with cols[1]:    
    mgroup = st.selectbox("Group",(groups), index=0) 


#query2 = "SELECT * FROM sovdb_schema.macro_indicators WHERE country = '"+countr+"' AND group = public"
#query2 = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""country"" = '"+countr+"' AND group = '"+groupm+"'"
if mgroup == 'all':
    query2 = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""country"" = '"+countr+"' AND ""mgroup"" IN "+tot_str
    #query2 = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""country"" = '"+countr+"' AND ""mgroup"" IN ('real','external')"
else:
    query2 = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""country"" = '"+countr+"' AND ""mgroup"" = '"+mgroup+"'"

#st.write(query2)
#st.write(len(groupm))
cur = conn.cursor()
cur.execute(query2);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
st.table(df[['ticker', 'full_name','freq','metric','mgroup']])