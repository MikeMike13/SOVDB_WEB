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

#conn.close()

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

def click_button_add_event(Start_date,Add_tags,Add_event,Add_long_event):    
    ticker = table_name
    query = "INSERT INTO sovdb_schema.\""+ticker+"\" (\"""Start_date\""", \"""tag\""", \"""des\""", \"""long_des\""") VALUES ('"+Start_date.strftime('%d-%b-%Y')+"'::date, '"+Add_tags+"':: text, '"+Add_event+"'::text, '"+Add_long_event+"'::text)"
    cur.execute(query)
    conn.commit()
    st.warning("New event was added")  
    st.session_state.clicked = True

def click_button_add_event_e(Start_date,End_date, Add_tags,Add_event,Add_long_event):    
    ticker = table_name
    query = "INSERT INTO sovdb_schema.\""+ticker+"\" (\"""Start_date\""", \"""tag\""", \"""des\""", \"""long_des\""",\"""End_date\""") VALUES ('"+Start_date.strftime('%d-%b-%Y')+"'::date, '"+Add_tags+"':: text, '"+Add_event+"'::text, '"+Add_long_event+"'::text, '"+End_date.strftime('%d-%b-%Y')+"'::date)"
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
    
#read all countries des
df_all = sovdb_read_gen("countries")
count_sel = df_all.name
df_all = df_all.fillna('') 

st.subheader("EVENTS")

cols=st.columns(3) 
with cols[0]:
    countr = st.selectbox("Country1",(count_sel), index=203)
    key = df_all[df_all.name==countr].m_key.values[0]
with cols[1]:
    countr2 = st.selectbox("Country2",(count_sel), index=203)
    key2 = df_all[df_all.name==countr2].m_key.values[0]
with cols[2]:
    countr3 = st.selectbox("Country3",(count_sel), index=203)
    key3 = df_all[df_all.name==countr3].m_key.values[0]

table_name = key+"_EVENTS"       
create_events_table = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+table_name+"\" ( ID SERIAL PRIMARY KEY, \"""Start_date\""" date, \"""End_date\""" date DEFAULT NULL, \"""tag\""" text, \"""des\""" text, \"""long_des\""" text)"    
cur = conn.cursor()
cur.execute(create_events_table)
conn.commit()
events = sovdb_read_gen(table_name)

table_name2 = key2+"_EVENTS"       
create_events_table2 = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+table_name2+"\" ( ID SERIAL PRIMARY KEY, \"""Start_date\""" date, \"""End_date\""" date DEFAULT NULL, \"""tag\""" text, \"""des\""" text, \"""long_des\""" text)"    
cur = conn.cursor()
cur.execute(create_events_table2)
conn.commit()
events2 = sovdb_read_gen(table_name2)

table_name3 = key3+"_EVENTS"       
create_events_table3 = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+table_name3+"\" ( ID SERIAL PRIMARY KEY, \"""Start_date\""" date, \"""End_date\""" date DEFAULT NULL, \"""tag\""" text, \"""des\""" text, \"""long_des\""" text)"    
cur = conn.cursor()
cur.execute(create_events_table3)
conn.commit()
events3 = sovdb_read_gen(table_name3)

tags_events_all = []
for event in events.tag:
    x = event.split(',')
    x = [a.strip() for a in x]
    for x0 in x:
        tags_events_all.append(x0)

tags_events_all2 = []
for event in events2.tag:
    x = event.split(',')
    x = [a.strip() for a in x]
    for x0 in x:
        tags_events_all2.append(x0)
        
tags_events_all3 = []
for event in events3.tag:
    x = event.split(',')
    x = [a.strip() for a in x]
    for x0 in x:
        tags_events_all3.append(x0)
        
cols=st.columns(3) 
with cols[0]:
    tags = pd.Series(tags_events_all).unique()    
    #st.write(tags)
    #tags_n=np.where(tags=='president')[0][0].astype(int)
    tags=pd.Series(tags) 
    tags_n = tags[tags=='president'].index[0]
    #st.write(tags_n)
    tags_events = st.selectbox("tag1: ",tags, index=4)
with cols[1]:
    tags_events2 = st.selectbox("tag2: ",pd.Series(tags_events_all2).unique(), index=5)
with cols[2]:
    tags_events3 = st.selectbox("tag3: ",pd.Series(tags_events_all3).unique(), index=0)
    

tag_ind = []
for i in range(events.shape[0]):
    if(tags_events in events.loc[i].tag):
        tag_ind.append(float(events.loc[i].id))
events_f = events[(events.id.isin(tag_ind))].sort_values(by=['Start_date'], ascending=True).reset_index()


tag_ind2 = []
for i in range(events2.shape[0]):
    if(tags_events2 in events2.loc[i].tag):
        tag_ind2.append(float(events2.loc[i].id))
events_f2 = events2[(events2.id.isin(tag_ind2))].sort_values(by=['Start_date'], ascending=True).reset_index()

tag_ind3 = []
for i in range(events3.shape[0]):
    if(tags_events3 in events3.loc[i].tag):
        tag_ind3.append(float(events3.loc[i].id))
events_f3 = events3[(events3.id.isin(tag_ind3))].sort_values(by=['Start_date'], ascending=True).reset_index()


items = []    
for i in range(events_f.shape[0]):    
    if (events_f.loc[i].End_date):
        items.append({"id":str(events_f.loc[i].id),"content": events_f.loc[i].des, "start": events_f.loc[i].Start_date.strftime('%Y-%b-%d'), "end": events_f.loc[i].End_date.strftime('%Y-%b-%d'),"group":"1"})   
    else:
        items.append({"id":str(events_f.loc[i].id),"content": events_f.loc[i].des, "start": events_f.loc[i].Start_date.strftime('%Y-%b-%d'),"group":"1"})   

for i in range(events_f2.shape[0]):    
    if (events_f2.loc[i].End_date):
        items.append({"id":str(events_f2.loc[i].id),"content": events_f2.loc[i].des, "start": events_f2.loc[i].Start_date.strftime('%Y-%b-%d'), "end": events_f2.loc[i].End_date.strftime('%Y-%b-%d'),"group":"2"})   
    else:
        items.append({"id":str(events_f2.loc[i].id),"content": events_f2.loc[i].des, "start": events_f2.loc[i].Start_date.strftime('%Y-%b-%d'),"group":"2"})   

for i in range(events_f3.shape[0]):    
    if (events_f3.loc[i].End_date):
        items.append({"id":str(events_f3.loc[i].id),"content": events_f3.loc[i].des, "start": events_f3.loc[i].Start_date.strftime('%Y-%b-%d'), "end": events_f3.loc[i].End_date.strftime('%Y-%b-%d'),"group":"3"})   
    else:
        items.append({"id":str(events_f3.loc[i].id),"content": events_f3.loc[i].des, "start": events_f3.loc[i].Start_date.strftime('%Y-%b-%d'),"group":"3"})   



#st.write(items)
groups = [
    {"id": 1, "content": tags_events, "style": "color: black; background-color: #a9a9a98F;"},
    {"id": 2, "content": tags_events2, "style": "color: black; background-color: #a9a9a98F;"},
    {"id": 3, "content": tags_events3, "style": "color: black; background-color: #a9a9a98F;"}
]
timeline = st_timeline(items, groups=groups, options={"verticalScroll": True,"height": 600,"margin": {"axis": 5}}, height="300px")


if (timeline!=None):
    sel_id = timeline.get('id')
    selected_event = sovdb_read_item(table_name, 'id', sel_id)
    
    st.write(selected_event[1].strftime('%d-%b-%Y'))
    st.write(selected_event[3])
    st.write(selected_event[4])
    
st.subheader("Add event")
cols=st.columns(3) 
with cols[0]:
    event_date = st.date_input("Event date: ", pd.to_datetime('2000-12-31'))
with cols[1]:
    is_end = st.checkbox('is end', 0) 
    if (is_end):
        end_date = st.date_input("End date: ", pd.to_datetime('2000-12-31'))
with cols[2]:
    tags = st.text_input('Tags (, separate)', '')


des = st.text_input('event, title', '')   
long_des = st.text_input('event, des', '')  

    
cols=st.columns(5)    
with cols[0]: 
    if is_end:
        st.button('Add event', on_click=click_button_add_event_e, args=(event_date,end_date,tags,des,long_des)) 
    else:
        st.button('Add event', on_click=click_button_add_event, args=(event_date,tags,des,long_des)) 
with cols[1]:
    event_id_del = st.selectbox("id: ",events_f.id, index=0)
with cols[2]:
    st.button('Show event', on_click=click_button_delete_event, args=(1,event_id_del)) 
with cols[3]:
    st.button('Update event', on_click=click_button_delete_event, args=(1,event_id_del))     
with cols[4]:
    st.button('Delete event', on_click=click_button_delete_event, args=(1,event_id_del)) 
        
st.write(events_f)