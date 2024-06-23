import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
import matplotlib 
import matplotlib.pyplot as plt

st.set_page_config(layout="centered")

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)
##DURATION
##https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/TQCB/securities/RU000A1026B3.xml?from=2024-04-01&till=2024-04-11

mymap = ["#0051CA", "#F8AC27", "#3F863F", "#C6DBA1", "#FDD65F", "#FBEEBD", "#50766E"];

st.title('1. Show issuer') 
#ticker_show = st.text_input('ticker','RU_EXPG_M_CBR')
#st.write('Description')
#query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_show+"'"    
#cur = conn.cursor()
#cur.execute(query);
#rows = cur.fetchall()
#colnames = [desc[0] for desc in cur.description]
#ticker_sel = pd.DataFrame(rows,columns=colnames)
    
#cols=st.columns(1)

#st.dataframe(
#    ticker_sel.T,
#    hide_index=False,
#    column_config={
#        #"col0": None,
#        #"col0": st.column_config.TextColumn(label="field"),
#        "col0": st.column_config.TextColumn(label="Value", width="large"),        
#    },
#    width=1200,
#    height=1550,
#)

st.title('2. Edit issuer') 
st.title('3. New issuer') 