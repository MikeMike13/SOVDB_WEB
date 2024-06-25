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

def sovdb_read_gen(ticker):
    selectquery = "SELECT * FROM sovdb_schema.\""+ticker+"\""
    cur = conn.cursor()
    cur.execute(selectquery);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows,columns=colnames)     
    return df

def sovdb_read_des_iss(tbl, ticker):
    #get des of selected bond
    query_des = "SELECT * FROM sovdb_schema."+tbl+" WHERE ""issuer_id""='"+str(ticker)+"'"
    #st.write(query_des)
    cur = conn.cursor()
    cur.execute(query_des);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    df_des = pd.DataFrame(rows,columns=colnames)
    return df_des

##DURATION
##https://iss.moex.com/iss/history/engines/stock/markets/bonds/boards/TQCB/securities/RU000A1026B3.xml?from=2024-04-01&till=2024-04-11

mymap = ["#0051CA", "#F8AC27", "#3F863F", "#C6DBA1", "#FDD65F", "#FBEEBD", "#50766E"];

iss = sovdb_read_gen("issuers")

st.title('1. Show issuer') 
st.write("Issuers count: "+str(iss.shape[0]))
iss_sel = st.selectbox("Issuer: ",(iss['rus_long'].sort_values(ascending=True)), index=1)  
idd = iss[iss['rus_long']==iss_sel].issuer_id.values[0]
#st.write(sovdb_read_des_iss("issuers", idd).T)
st.dataframe(
    sovdb_read_des_iss("issuers", idd).T,
    hide_index=False,
    column_config={
        #"col0": None,
        #"col0": st.column_config.TextColumn(label="field"),
        "col0": st.column_config.TextColumn(label="Value", width="large"),        
    },
    width=1200,
    height=700,
)
    
st.title('2. Edit issuer') 
iss_sel_e = st.selectbox("Choose issuer: ",(iss['rus_long'].sort_values(ascending=True)), index=1)  
idd_e = iss[iss['rus_long']==iss_sel].issuer_id.values[0]

cols=st.columns(5)
with cols[0]:
    issuer_id_moex_e = st.text_input('moex id: ', key="issuer_id_e_moex_def")
with cols[1]:
    inn_e = st.text_input('INN: ', key="inn_e_def")
with cols[2]:
    okpo_e = st.text_input('OKPO: ', key="okpo_e_def")
with cols[3]:
    rus_long_e = st.text_input("RUS long: ", key="rus_long_e_def")
with cols[4]:
    rus_short_e = st.text_input("RUS short: ", key="rus_short_e_def")
    
cols=st.columns(5)
with cols[0]:
    eng_long_e = st.text_input('ENG long: ', key="ieng_long_e_def")
with cols[1]:
    eng_short_e = st.text_input('ENG short: ', key="eng_short_e_def")
with cols[2]:
    industry_e = st.text_input('Industry: ', key="industry_e_def")
with cols[3]:
    industry_bics_e = st.text_input("Industry (bloom): ", key="industry_bics_e_def")
with cols[4]:
    industry_gics_e = st.text_input("Industry (S&P, MCSI): ", key="industry_gics_e_def")

cols=st.columns(5)
with cols[0]:
    industry_isic_e = st.text_input('Industry (UN): ', key="industry_isic_e_def")
with cols[1]:
    industry_cis_e = st.text_input('Industry (US): ', key="industry_cis_e_def")
with cols[2]:
    industry_trbc_e = st.text_input('Industry (Refinitiv): ', key="industry_trbc_e_def")
with cols[3]:
    web_e = st.text_input("Site: ", key="web_e_def")
with cols[4]:
    reports_e = st.text_input("Fin reports: ", key="reports_e_def")

    
cols=st.columns(5)
with cols[0]:
    country_regestered_e = st.text_input('Country (regestered): ', key="country_regestered_e_def")
with cols[1]:
    country_e = st.text_input('Country (origin): ', key="country_e_def")
with cols[2]:
    parent_id_e = st.text_input('Parent company id: ', key="parent_id_e_def")


    
st.title('3. New issuer') 