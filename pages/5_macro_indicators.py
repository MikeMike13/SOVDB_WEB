import streamlit as st
import psycopg2 as ps
import pandas as pd
#import math
from datetime import date
#import openpyxl
import matplotlib 
import matplotlib.pyplot as plt
import numpy as np

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

mymap = ['#0051CA', '#F8AC27', '#3F863F', '#C6DBA1', '#FDD65F', '#FBEEBD', '#50766E'];

st.title('New indicator')
if "ticker_def" not in st.session_state:
    st.session_state.ticker_def = ''
if "alive_def" not in st.session_state:
    st.session_state.alive_def = 1
#if "country_def" not in st.session_state:
#    st.session_state.country_def = ''
if "short_name_def" not in st.session_state:
    st.session_state.short_name_def = ''
if "short_name_rus_def" not in st.session_state:
    st.session_state.short_name_rus_def = ''
if "full_name_def" not in st.session_state:
    st.session_state.full_name_def = ''
if "full_name_rus_def" not in st.session_state:
    st.session_state.full_name_rus_def = ''
if "label_def" not in st.session_state:
    st.session_state.label_def = ''
if "label_rus_def" not in st.session_state:
    st.session_state.label_rus_def = ''
if "metric_def" not in st.session_state:
    st.session_state.metric_def = ''
if "mult_def" not in st.session_state:
    st.session_state.mult_def = 0
if "sa_def" not in st.session_state:
    st.session_state.sa_def = 0
if "ca_def" not in st.session_state:
    st.session_state.ca_def = 0
if "freq_def" not in st.session_state:
    st.session_state.freq_def = 'D'
if "eop_avg_def" not in st.session_state:
    st.session_state.eop_avg_def = ''
    
if "stock_flow_def" not in st.session_state:
    st.session_state.stock_flow_def = ''
if "real_nominal_def" not in st.session_state:
    st.session_state.real_nominal_def = ''    
if "group_def" not in st.session_state:
    st.session_state.group_def = ''
if "subgroup_def" not in st.session_state:
    st.session_state.subgroup_def = ''
if "subsubgroup_def" not in st.session_state:
    st.session_state.subsubgroup_def = ''

if "subsubsubgroup_def" not in st.session_state:
    st.session_state.subsubsubgroup_def = ''
if "customgroup_def" not in st.session_state:
    st.session_state.customgroup_def = ''    
if "opendata_def" not in st.session_state:
    st.session_state.opendata_def = 1
if "methodology_name_def" not in st.session_state:
    st.session_state.methodology_name_def = ''
if "meth_raw_indic_def" not in st.session_state:
    st.session_state.meth_raw_indic_def = ''

if "meth_custom_indic_def" not in st.session_state:
    st.session_state.meth_custom_indic_def = ''
if "meth_link_def" not in st.session_state:
    st.session_state.meth_link_def = ''    
if "source_org_def" not in st.session_state:
    st.session_state.source_org_def = ''
if "source_db_def" not in st.session_state:
    st.session_state.source_db_def = ''
if "db_code_def" not in st.session_state:
    st.session_state.db_code_def = ''
    
if "db_link_def" not in st.session_state:
    st.session_state.db_link_def = ''
if "next_indic_def" not in st.session_state:
    st.session_state.next_indic_def = ''    
if "prev_indic_def" not in st.session_state:
    st.session_state.prev_indic_def = ''
if "start_date_def" not in st.session_state:
    st.session_state.start_date_def = date.today()
if "end_date_def" not in st.session_state:
    st.session_state.end_date_def = date.today()
    
if "next_update_def" not in st.session_state:
    st.session_state.next_update_def = date.today()
if "update_date_def" not in st.session_state:
    st.session_state.update_date_def = date.today()    
if "first_down_date_def" not in st.session_state:
    st.session_state.first_down_date_def = date.today()
if "custom_def" not in st.session_state:
    st.session_state.custom_def = 0
if "dima_id_def" not in st.session_state:
    st.session_state.dima_id_def = ''
     
if "formula_def" not in st.session_state:
    st.session_state.formula_def = ''
if "des_def" not in st.session_state:
    st.session_state.des_def = ''
     
def paste_button(PASTE, ticker):
    st.session_state.alive_def = ticker.alive.values[0]
    #st.session_state.country_def = ticker.country.values[0]
    st.session_state.short_name_def     = ticker.short_name.values[0]
    st.session_state.short_name_rus_def = ticker.short_name_rus.values[0]
    st.session_state.full_name_def     = ticker.full_name.values[0]
    st.session_state.full_name_rus_def = ticker.full_name_rus.values[0]
    st.session_state.label_def         = ticker.label.values[0]
    st.session_state.label_rus_def     = ticker.label_rus.values[0]
    st.session_state.metric_def        = ticker.metric.values[0]
    
    st.session_state.mult_def          = ticker.mult.values[0]
    st.session_state.sa_def            = ticker.sa.values[0]
    st.session_state.ca_def            = ticker.ca.values[0]
    st.session_state.freq_def          = ticker.freq.values[0]
    st.session_state.eop_avg_def       = ticker.eop_avg.values[0]

    st.session_state.stock_flow_def    = ticker.stock_flow.values[0]
    st.session_state.real_nominal_def  = ticker.real_nominal.values[0]
    st.session_state.group_def         = ticker.mgroup.values[0]
    st.session_state.subgroup_def      = ticker.subgroup.values[0]
    st.session_state.subsubgroup_def   = ticker.subsubgroup.values[0]
    
    st.session_state.subsubsubgroup_def   = ticker.subsubsubgroup.values[0]
    st.session_state.customgroup_def      = ticker.customgroup.values[0]
    st.session_state.opendata_def         = ticker.opendata.values[0]
    st.session_state.methodology_name_def = ticker.methodology_name.values[0]
    st.session_state.meth_raw_indic_def   = ticker.meth_raw_indic.values[0]
    
    st.session_state.meth_custom_indic_def = ticker.meth_custom_indic.values[0]
    st.session_state.meth_link_def         = ticker.meth_link.values[0]
    st.session_state.source_org_def        = ticker.source_org.values[0]
    st.session_state.source_db_def         = ticker.source_db.values[0]
    st.session_state.db_code_def           = ticker.db_code.values[0]

    st.session_state.db_link_def           = ticker.db_link.values[0]
    st.session_state.next_indic_def        = ticker.next_indic.values[0]
    st.session_state.prev_indic_def        = ticker.prev_indic.values[0]
    st.session_state.start_date_def        = ticker.start_date.values[0]
    st.session_state.end_date_def          = ticker.end_date.values[0]

    st.session_state.next_update_def      = ticker.next_update.values[0]
    st.session_state.update_date_def      = ticker.update_date.values[0]
    st.session_state.first_down_date_def  = ticker.first_down_date.values[0]
    st.session_state.custom_def           = ticker.custom.values[0]
    st.session_state.dima_id_def          = ticker.dima_id.values[0]

    st.session_state.formula_def          = ticker.formula.values[0]
    st.session_state.des_def              = ticker.des.values[0]
    
cols=st.columns(2)
with cols[0]:
    ticker_src = st.text_input('Copy des from ticker: ', 'RU_CBRLIQFORMCBCORR_D_CBR')
    
    query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_src+"'"    
    cur = conn.cursor()
    cur.execute(query);
    rows = cur.fetchall()
    colnames = [desc[0] for desc in cur.description]
    ticker_sel = pd.DataFrame(rows,columns=colnames)
#    st.write(ticker_sel.short_name)
with cols[1]:        
    PASTE = 1
    st.button('Paste', on_click=paste_button, args=(PASTE, ticker_sel)) 
   
cols=st.columns(5)
with cols[0]:
    ticker_n = st.text_input('ticker','', key="ticker_def")
    #ticker = st.text_input('Ticker', 'FX_RUBUSD_CBR')
with cols[1]:
    alive = st.checkbox('alive', key="alive_def")                          
with cols[2]:
    country = st.selectbox('country', (count_sel), index=203)
with cols[3]:
    short_name = st.text_input('short name', '', key="short_name_def")
with cols[4]:
    short_name_rus = st.text_input('short name rus', '', key="short_name_rus_def")

cols=st.columns(5)
with cols[0]:
    full_name = st.text_input('full name', '', key="full_name_def")
with cols[1]:
    full_name_rus = st.text_input('full name rus', '', key="full_name_rus_def")                
with cols[2]:
    label = st.text_input('label', '', key="label_def")
with cols[3]:
    label_rus = st.text_input('label rus', '', key="label_rus_def")
with cols[4]:
    metric = st.text_input('metric', '', key="metric_def")

cols=st.columns(5)
with cols[0]:
    mult = st.number_input('mult',key="mult_def")
with cols[1]:
    sa = st.checkbox('sa',key="sa_def")     
with cols[2]:
    ca = st.checkbox('ca',key="ca_def")     
with cols[3]:
    freq = st.selectbox('freq', ('D','M','Q','Y'), index=0,key="freq_def")
with cols[4]:
    eop_avg = st.text_input('eop/avg', key="eop_avg_def")
    
cols=st.columns(5)
with cols[0]:
    stock_flow = st.text_input('stock/flow', key="stock_flow_def")
with cols[1]:
    real_nominal = st.text_input('real/nominal', key="real_nominal_def")
with cols[2]:
    group = st.text_input('group', key="group_def")
with cols[3]:
    subgroup = st.text_input('subgroup', key="subgroup_def")
with cols[4]:
    subsubgroup = st.text_input('subsubgroup', key="subsubgroup_def")
    
cols=st.columns(5)
with cols[0]:
    subsubsubgroup = st.text_input('subsubsubgroup', key="subsubsubgroup_def")
with cols[1]:
    customgroup = st.text_input('customgroup', key="customgroup_def")
with cols[2]:
    opendata = st.checkbox('opendata', key="opendata_def")
with cols[3]:
    methodology_name = st.text_input('methodology_name', key="methodology_name_def")
with cols[4]:
    meth_raw_indic = st.text_input('meth raw indic', key="meth_raw_indic_def")
    
cols=st.columns(5)
with cols[0]:
    meth_custom_indic = st.text_input('meth custom indic', key="meth_custom_indic_def")
with cols[1]:
    meth_link = st.text_input('meth link', key="meth_link_def")
with cols[2]:
    source_org = st.text_input('source org', key="source_org_def")
with cols[3]:
    source_db = st.text_input('source db', key="source_db_def")
with cols[4]:
    db_code = st.text_input('db code', key="db_code_def")
        
cols=st.columns(5)
with cols[0]:
    db_link = st.text_input('db link', key="db_link_def")
with cols[1]:
    next_indic = st.text_input('next indic', key="next_indic_def")
with cols[2]:
    prev_indic = st.text_input('prev indic', key="prev_indic_def")
with cols[3]:
    start_date = st.date_input("Start date: ", key="start_date_def")
with cols[4]:
    end_date = st.date_input("End date: ", key="end_date_def")
        
cols=st.columns(5)
with cols[0]:
    next_update = st.date_input("next update: ", key="next_update_def")
with cols[1]:
    update_date = st.date_input("update date: ", key="update_date_def")
with cols[2]:
    first_down_date = st.date_input("first down date: ", key="first_down_date_def")
with cols[3]:
    #custom = st.number_input('custom', key="custom_def")
    custom = st.checkbox('custom', key="custom_def")
with cols[4]:
    dima_id = st.text_input("dima id ", key="dima_id_def" )
     
def create_button():
    new_indic_des = "INSERT INTO sovdb_schema.""macro_indicators"" ("\
    "\"""ticker\""", \"""alive\""", \"""country\""", \"""short_name\""", \"""short_name_rus\""", \"""full_name\""", \"""full_name_rus\""", \"""label\""", \"""label_rus\""", \"""metric\""", \"""mult\""", \"""sa\""", \"""ca\""", \"""freq\""", \"""eop_avg\""", \"""stock_flow\""", "\
    "\"""real_nominal\""", \"""mgroup\""", \"""subgroup\""", \"""subsubgroup\""", \"""subsubsubgroup\""", \"""customgroup\""", \"""opendata\""", \"""methodology_name\""", \"""meth_raw_indic\""", \"""meth_custom_indic\""", \"""meth_link\""", "\
    "\"""source_org\""", \"""source_db\""", \"""db_code\""", \"""db_link\""", \"""next_indic\""", \"""prev_indic\""", \"""start_date\""", \"""end_date\""", \"""next_update\""", \"""update_date\""", \"""first_down_date\""", \"""custom\""", \"""formula\""", \"""dima_id\""", "\
    "\"""des\""") VALUES ("\
    "'"+ticker_n+"'::text, '"+str(alive)+"'::boolean, '"+country+"'::text, '"+short_name+"'::text, '"+short_name_rus+"'::text, '"+full_name+"'::text, '"+full_name_rus+"'::text, '"\
    +label+"'::text, '"+label_rus+"'::text, '"+metric+"'::text, '"+str(mult)+"'::double precision, '"+str(sa)+"'::boolean, '"+str(ca)+"'::boolean, '"+freq+"'::text, '"\
    +eop_avg+"'::text, '"+stock_flow+"'::text, '"+real_nominal+"'::text, '"+group+"'::text, '"+subgroup+"'::text, '"+subsubgroup+"'::text, '"+subsubsubgroup+"'::text, '"\
    +customgroup+"'::text, '"+str(opendata)+"'::boolean, '"+methodology_name+"'::text, '"+meth_raw_indic+"'::text, '"+meth_custom_indic+"'::text, '"+meth_link+"'::text, '"\
    +source_org+"'::text, '"+source_db+"'::text, '"+db_code+"'::text, '"+db_link+"'::text, '"+next_indic+"'::text, '"+prev_indic+"'::text, '"+start_date.strftime('%d-%b-%Y')+"'::date, '"\
    +end_date.strftime('%d-%b-%Y')+"'::date, '"+next_update.strftime('%d-%b-%Y')+"'::date, '"+update_date.strftime('%d-%b-%Y')+"'::date, '"+first_down_date.strftime('%d-%b-%Y')+"'::date, '"+str(custom)+"'::boolean, '"\
    +formula+"'::text, '"+dima_id+"'::text, '"+des+"'::text) returning \"""ticker\""";"
        
    #st.write(new_indic_des)
    cur.execute(new_indic_des)
    conn.commit()
    st.warning("NEW INDICATOR DES: "+ticker_n+" WAS ADDED")            
    
    new_indic_data = "CREATE TABLE IF NOT EXISTS sovdb_schema.\""+ticker_n+"\" ( \"""Date\""" date, \"""Value\""" double precision)"    
    #st.write(new_indic_data)
    cur.execute(new_indic_data)
    conn.commit()
    st.warning("NEW INDICATOR DATA: "+ticker_n+" WAS ADDED")        
    
def delete_button():
    del_indic_des = "DELETE FROM sovdb_schema.""macro_indicators"" WHERE \"""ticker\""" = '"+ticker_n+"'"        
    cur.execute(del_indic_des)
    conn.commit()
    st.warning("INDICATOR DES: "+ticker_n+" WAS DELETED")
        
    del_indic_data = "DROP TABLE IF EXISTS sovdb_schema.\""+ticker_n+"\""       
    cur.execute(del_indic_data)
    conn.commit()
    st.warning("INDICATOR DATA: "+ticker_n+" WAS DELETED")
    
cols=st.columns(5)
with cols[0]:
    formula = st.text_input("formula: ", key="formula_def" )
with cols[1]:
    des = st.text_input("des: ", key="des_def"  )
with cols[2]:    
    st.button('Create indicator', on_click=create_button)
    st.write(ticker_n)
with cols[3]:
    st.button('Delete indicator', on_click=delete_button)
    st.write(ticker_n)
    
st.title('Edit indicator') 
                        
st.title('Show indicator des') 
ticker_show = st.text_input('ticker','')
query = "SELECT * FROM sovdb_schema.""macro_indicators"" WHERE ""ticker"" = '"+ticker_show+"'"    
cur = conn.cursor()
cur.execute(query);
rows = cur.fetchall()
colnames = [desc[0] for desc in cur.description]
ticker_sel = pd.DataFrame(rows,columns=colnames)


#cols=st.columns(1)

st.dataframe(
    ticker_sel.T,
    hide_index=False,
    column_config={
        #"col0": None,
        #"col0": st.column_config.TextColumn(label="field"),
        "col0": st.column_config.TextColumn(label="Value", width="large"),        
    },
    width=1200,
    height=1550,
)

st.title('Upload data') 

cols=st.columns(4)
with cols[0]:
    list_xls = st.text_input("list: ",'EXPIMP_M')
with cols[1]:
    ind_col = st.number_input("Tickers col: ",value=2)
with cols[2]:
    start_col = st.number_input("Column from: ",value=16)
with cols[3]:
    end_col = st.number_input("Column to: ",value=208)

    
uploaded_file = st.file_uploader("Choose a file", type = 'xlsx')
if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file, sheet_name=list_xls, decimal ='.',)
    
    get_ind = df_raw.iloc[:,ind_col].dropna().index
    
    df_raw_sel = df_raw.iloc[get_ind,start_col-1:end_col]    
    colnames = df_raw.iloc[get_ind,ind_col-1]
    df_new = pd.DataFrame(df_raw_sel.values.T,columns=colnames)
    df_new = pd.DataFrame(df_new).set_index('Date')
    df_new.index = pd.to_datetime(df_new.index)
    
    data_old = []  
     
    no_data = 0
    j=0
    for indics in colnames[1:]:
        #st.write(indics)
        
        query = "SELECT * FROM sovdb_schema.\""+indics+"\"  ORDER BY \"""Date\""" ASC"
        #cur = conn.cursor()
        cur.execute(query);
        rows = cur.fetchall()
        rows = np.array([*rows])
        index_old = []
        if rows.size==0:
            #st.write(rows)
            no_data = 1
            #st.write('no data')
        else:
            #colnames = [desc[0] for desc in cur.description]
            #df = pd.DataFrame(rows,columns=colnames)
            #df = pd.DataFrame(df).set_index('Date')
            #df = df.sort_index()        
            #df.index = pd.to_datetime(df.index)                
            #st.write(rows[:,1])
            data_old.insert(j, rows[:,1].tolist())
            j=j+1
            index_old = pd.to_datetime(rows[:,0].tolist())
            #rows[:,1]
        
    if no_data:
        df_comb = df_new
        df_old = pd.DataFrame([],columns=colnames)
        st.write('indicators are downloaded for the first time')
    else:
        #st.write(data_old)        
        #st.write(colnames)
        df_old = pd.DataFrame(np.array(data_old).T.tolist(),columns=colnames[1:],index=index_old)
        df_old = df_old.sort_index() 
        df_comb = pd.concat([df_old, df_new], ignore_index=False, sort=True)
        df_comb = df_comb[df_comb.index>=df_new.index[0]]
        df_comb = df_comb.sort_index() 
    
    cols=st.columns(3)
    with cols[0]:
       st.write('Is in the database')
       st.write(df_old)
    with cols[1]:
       st.write('New data') 
       st.write(df_new)
    with cols[2]:
       st.write('Combined data to download') 
       st.write(df_comb)
    
    new_indics_plot = st.selectbox('Select indicator to plot', (colnames[1:]), index=0)
    
    fig, ax = plt.subplots()    
    ax.plot(df_old[new_indics_plot], color=mymap[0], linestyle = '--', label='in database',linewidth=0.8) 
    ax.plot(df_new[new_indics_plot], color=mymap[0], label='new data',linewidth=0.8) 
    ax.plot(df_comb[new_indics_plot], color=mymap[1], linestyle = ':', label='combined',linewidth=0.8) 
    
        
    plt.legend() 
    formatter = matplotlib.dates.DateFormatter('%Y')
    ax.xaxis.set_major_formatter(formatter)
    plt.show() 
    st.pyplot(fig)
    
    INS=1    
    def write_indicator_to_sovdb(INS, df_comb2):
        for indic in df_comb2.columns:
            #st.write(indic)
            #st.write(df_comb2[indic].size)
            #st.write(df_comb2[indic][0], df_comb2.index[0])
            
            #delete all data from indicator table
            for dates_del in df_comb2.index:
                delete_q = "DELETE FROM sovdb_schema.\""+indic+"\" WHERE \"""Date\"""= '"+dates_del.strftime('%d-%b-%Y')+"'"
                cur.execute(delete_q)
                conn.commit()                              
                        
            for value in range(df_comb2[indic].size):
                query = "INSERT INTO sovdb_schema.\""+indic+"\" (\"""Date\""", \"""Value\""") VALUES ('"+df_comb2.index[value].strftime('%d-%b-%Y')+"'::date, '"+str(df_comb2[indic][value])+"':: double precision) returning \"""Date\""""
                cur.execute(query)
                conn.commit()
                #st.write(df_comb2[indic][value-1], df_comb2.index[value-1])
                 
            #update dates in des            
            TD = date.today()
            if no_data:
                query = "UPDATE sovdb_schema.""macro_indicators"" SET \"""start_date\""" = '"+str(df_comb2.index[0].strftime('%d-%b-%Y'))+"', \"""end_date\""" = '"+str(df_comb2.index[-1].strftime('%d-%b-%Y'))+"', \"""update_date\""" = '"+str(TD.strftime('%d-%b-%Y'))+"',\"""first_down_date\""" = '"+str(TD.strftime('%d-%b-%Y'))+"' WHERE \"""ticker\""" = '"+indic+"'"        
                st.warning(indic+" DATA IS ADDED")
            else:
                query = "UPDATE sovdb_schema.""macro_indicators"" SET \"""start_date\""" = '"+str(df_comb2.index[0].strftime('%d-%b-%Y'))+"', \"""end_date\""" = '"+str(df_comb2.index[-1].strftime('%d-%b-%Y'))+"', \"""update_date\""" = '"+str(TD.strftime('%d-%b-%Y'))+"' WHERE \"""ticker\""" = '"+indic+"'"        
                st.warning(indic+" DATA IS UPDATED")
            cur.execute(query)
            conn.commit()
        
    

    st.button('Add new data to sovdb', on_click=write_indicator_to_sovdb, args=(INS, df_comb))
    
    
    