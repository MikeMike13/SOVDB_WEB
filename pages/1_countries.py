import streamlit as st
import psycopg2 as ps
import pandas as pd
import math

conn = ps.connect(database = "sovdb", 
                        user = "mike", 
                        host= '185.26.120.148',
                        password = "mikesovdb13",
                        port = 5432)




query = "SELECT * FROM sovdb_schema.countries"
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
df = pd.DataFrame(rows,columns=colnames)
count_sel = df.name

df = df.fillna('') 

countr = st.selectbox("Country",(count_sel), index=203)
#st.write(df)
#st.write(df[df.name==countr].type.values[0])

cols=st.columns(2)
with cols[0]:
    if (len(df[df.name==countr].type.values[0])==0):
        st.write("Type: none")
    else:
        st.write("Type: "+df[df.name==countr].type.values[0])
with cols[1]:
    if (len(df[df.name==countr].fx.values[0])==0):
        st.write("Currency: none")
    else:
        st.write("Currency: "+df[df.name==countr].fx.values[0])    
    
cols=st.columns(4)
with cols[0]:
    st.write("Master key (M_KEY): "+df[df.name==countr].m_key.values[0])
with cols[1]:
    if (len(df[df.name==countr].imf_key.values[0])==0):
        st.write("IMF key: none")
    else:
        st.write("IMF key: "+df[df.name==countr].imf_key.values[0])
with cols[2]:
    if (len(df[df.name==countr].wb_key.values[0])==0):
        st.write("WB key: none")
    else:
        st.write("WB key: "+df[df.name==countr].wb_key.values[0])      
with cols[3]:    
    if (len(df[df.name==countr].ecb_key.values[0])==0):
        st.write("ECB key: none")
    else:
        st.write("ECB key: "+df[df.name==countr].ecb_key.values[0])

cols=st.columns(4)
with cols[0]:
    if (len(df[df.name==countr].oecd_key.values[0])==0):
        st.write("OECD key: none")
    else:
        st.write("OECD key: "+df[df.name==countr].oecd_key.values[0])
with cols[1]:
    if (len(df[df.name==countr].bis_key.values[0])==0):
        st.write("BIS key: none")
    else:
        st.write("BIS key: "+df[df.name==countr].bis_key.values[0])
with cols[2]:
    if (len(df[df.name==countr].iso2_key.values[0])==0):
        st.write("ISO2 key: none")
    else:
        st.write("ISO2 key: "+df[df.name==countr].iso2_key.values[0])      
with cols[3]:    
    if (len(df[df.name==countr].iso3_key.values[0])==0):
        st.write("ISO3 key: none")
    else:
        st.write("ISO3 key: "+df[df.name==countr].iso3_key.values[0])