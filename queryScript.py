# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:18:23 2019

@author: tanya whyte
"""

# --------------------------------------------------------------------------
# Description
# --------------------------------------------------------------------------
'''
Script for extracting Hansard from SQL to DataFrame.

'''


import sqlalchemy as sqa
import pandas as pd

#From Tanya Whyte
engine = sqa.create_engine('postgresql://postgres:July11867@localhost:5432/lipad')
sql = sqa.text(' '.join((
    "SELECT *",
    "FROM dilipadsite_basehansard",
    "WHERE (pid != '' and pid != 'unmatched' and pid !='intervention')",
    "AND (speakerposition != 'subtopic' and speakerposition != 'topic' and speakerposition !='stagedirection')",
    "AND (speakername not like '%Member%')",
    "AND (speechtext <> '') ORDER BY basepk ;",
)))

data = pd.read_sql_query(sql, engine)
engine.dispose()



import pickle
data.to_pickle("./lipad.pkl")

dataShort = data.tail(100)
dataShort.to_pickle("./lipadShort.pkl")