import streamlit as st
import pandas as pd
import psycopg2 as ps
import numpy as np
from streamlit_timeline import st_timeline


st.set_page_config(layout="wide")


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

def sovdb_read_item(ticker, field, value):
    query = "SELECT * FROM sovdb_schema.\""+ticker+"\" WHERE \""+field+"\"='"+str(value)+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    rows = np.array([*rows])   
    if rows.size ==0:
        return 0
    else:
        return rows[0]
    
#read all countries des
df_all = sovdb_read_gen("countries")
count_sel = df_all.name
df_all = df_all.fillna('') 

countr = st.selectbox("Country",(count_sel), index=203)
key = df_all[df_all.name==countr].m_key.values[0]

st.subheader("EVENTS")
table_name = key+"_EVENTS"       
create_events_table = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+table_name+"\" ( ID SERIAL PRIMARY KEY, \"""Date\""" date, \"""tag\""" text, \"""des\""" text, \"""long_des\""" text)"    
cur = conn.cursor()
cur.execute(create_events_table)
conn.commit()

events = sovdb_read_gen(table_name)

tags_events_all = []
for event in events.tag:
    x = event.split(',')
    x = [a.strip() for a in x]
    for x0 in x:
        tags_events_all.append(x0)

tags_events = st.selectbox("choose tag: ",pd.Series(tags_events_all).unique(), index=0)

tag_ind = []
for i in range(events.shape[0]):
    if(tags_events in events.loc[i].tag):
        tag_ind.append(float(events.loc[i].id))
events_f = events[(events.id.isin(tag_ind))].sort_values(by=['Date'], ascending=True).reset_index()

items = []    
for i in range(events_f.shape[0]):
    items.append({"id":str(events_f.loc[i].id),"content": events_f.loc[i].des, "start": events_f.loc[i].Date.strftime('%Y-%b-%d')})   

#st.write(events_f)
timeline = st_timeline(items, groups=[], options={}, height="300px")


if (timeline!=None):
    sel_id = timeline.get('id')
    selected_event = sovdb_read_item(table_name, 'id', sel_id)
    
    st.write(selected_event[1].strftime('%d-%b-%Y'))
    st.write(selected_event[3])
    st.write(selected_event[4])
    
st.subheader("Add event")
cols=st.columns(2) 
with cols[0]:
    event_date = st.date_input("Event date: ", pd.to_datetime('2000-12-31'))
with cols[1]:
    tags = st.text_input('Tags (, separate)', '')

des = st.text_input('event, title', '')   
long_des = st.text_input('event, des', '')   

def click_button_add_event(Add_date,Add_tags,Add_event,Add_long_event):    
    ticker = table_name
    query = "INSERT INTO sovdb_schema.\""+ticker+"\" (\"""Date\""", \"""tag\""", \"""des\""", \"""long_des\""") VALUES ('"+Add_date.strftime('%d-%b-%Y')+"'::date, '"+Add_tags+"':: text, '"+Add_event+"'::text, '"+Add_long_event+"'::text)"
    cur.execute(query)
    conn.commit()
    st.warning("New event was added")  
    st.session_state.clicked = True

def click_button_delete_event(DEL,event_id_del):    
    ticker = table_name
    query = "DELETE FROM sovdb_schema.\""+ticker+"\" WHERE \"""id\"""='"+str(event_id_del)+"'"
    cur.execute(query)
    conn.commit()
    st.warning("Event was deleted")  
    st.session_state.clicked = True
    
cols=st.columns(3)    
with cols[0]: 
    st.button('Add event', on_click=click_button_add_event, args=(event_date,tags,des,long_des)) 
with cols[1]:
    event_id_del = st.selectbox("id to delete: ",events_f.id, index=0)
with cols[2]:
    st.button('Delete event', on_click=click_button_delete_event, args=(1,event_id_del)) 
    
st.write(events_f)