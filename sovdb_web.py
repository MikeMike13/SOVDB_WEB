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
        if FREQ == "M":
            if df.index[i].month == 1:
                de_cum.append(df[i])                  
            else:                
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
    if method=="EOP":
        df = df.resample(freq).last()
    elif method=="AVG":
        df = df.resample(freq).mean()
    elif method=="SUM":
        df = df.resample(freq).sum()
    elif method=="DECUM":
        df = decum(df,freq)
elif freq=="Q":
    if method=="EOP":
        df = df.resample('Q').last()
    elif method=="AVG":
        df = df.resample('Q').mean()
    elif method=="SUM":
        df = df.resample('Q').sum()
elif freq=="Y":
    if method=="EOP":
        df = df.resample('Y').last()
    elif method=="AVG":
        df = df.resample('Y').mean()
    elif method=="SUM":
        df = df.resample('Y').sum()           

if freq=="D":
    if trans=="mom":
        df.pct_change(periods=20) * 100
    elif trans=="qoq":
        df = df.pct_change(periods=3*20) * 100
    elif trans=="yoy":
        df = df.pct_change(periods=252) * 100 
elif freq=="M":
    if trans=="mom":
        df = df.pct_change(periods=1) * 100
    elif trans=="qoq":
        df = df.pct_change(periods=4) * 100
    elif trans=="yoy":
        df = df.pct_change(periods=12) * 100        
elif freq=="Q":
    if trans=="qoq":
        df = df.pct_change(periods=1) * 100
    elif trans=="yoy":
        df = df.pct_change(periods=4) * 100        
elif freq=="Y":
    if trans=="yoy":
        df = df.pct_change(periods=1) * 100        
            
if func=="avg":
    df = df.rolling(window=int(window_num)).mean()
elif func=="sum":
    df = df.rolling(int(window_num)).sum()
elif func=="std":
    df = df.rolling(int(window_num)).std().shift()
elif func=="cmlt":
    df = (1 + df/100).cumprod() - 1


#period change
cols=st.columns(4)
Start_val = 0
End_val = 0;
period_calc=''
with cols[0]:    
    bool_c = st.checkbox('Calculate returns')
with cols[1]:
    Start_date_c = st.date_input("From: ", datetime.date(2022, 1, 1))
    if bool_c:
        try:            
            Start_val = round(df.loc[pd.DatetimeIndex([Start_date_c])].values[0],2)       
            st.write(Start_val)
            period_calc = Start_val
        except:
            a=1  
with cols[2]:    
    End_date_c = st.date_input("To: ", date.today())
    if bool_c:
        try:            
            End_val = round(df.loc[pd.DatetimeIndex([End_date_c])].values[0],2)       
            st.write(End_val)
            period_calc = period_calc+" "+End_val
        except:
            a=1    
with cols[3]: 
    if bool_c and Start_val*End_val:
        period_ret = (End_val/Start_val-1)*100
        annula_ret = ((1+period_ret/100)**(365.25/(End_date_c - Start_date_c).days)-1)*100
        years = (End_date_c - Start_date_c).days/365.25
        
        st.write("abs: "+str(round(End_val-Start_val,2))+"; pct: "+str(round(period_ret,2))+"%") 
        st.write("ann ret: "+str(round(annula_ret,2))+"% ("+str(round(years,1))+"Y)")    
    
fig, ax = plt.subplots()
Lastdate = df.index[-1].strftime('%Y-%m-%d')
#st.write(colnames)
ax.plot(df, color=mymap[0], label='d',linewidth=0.8) 
#st.write(type(df))
ax.text(df.index[-1], df[-1], round(df[-1],2), fontsize=8,color=mymap[0]);

if bool_y:
    ax.axhline(y=y_level, color=(0.15, 0.15, 0.15), linestyle='-',linewidth=0.75)
#if bool_c and period_calc != 'no start value' and period_calc != 'no end value':
if bool_c and Start_val*End_val:    
    ax.plot(Start_date_c, Start_val, marker=5,color=(1,0,0)) 
    ax.plot(End_date_c, End_val, marker=4,color=(1,0,0)) 
    
    
plt.title(ticker+", "+Lastdate) 
plt.legend() 

formatter = matplotlib.dates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(formatter)

plt.show() 

st.pyplot(fig)

cols=st.columns(3)
with cols[0]:    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:    
        df.to_excel(writer, sheet_name='Sheet1', index=True)    
    download2 = st.download_button(
        label="Excel",
        data=buffer,
        file_name=ticker+".xlsx",
        mime='application/vnd.ms-excel'
    )
with cols[1]:    
  #  @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')
    
    csv = convert_df(df)
    
    st.download_button(
        label="CSV",
        data=csv,
        file_name=ticker+".csv",
        mime='text/csv',
    )
with cols[2]:
    fn = ticker+".png"
    plt.savefig(fn)
    with open(fn, "rb") as img:
        btn = st.download_button(
            label="JPG",
            data=img,
            file_name=fn,
            mime="image/png"
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