
# coding: utf-8

# In[8]:


import pandas as pd
import preprocess_picanet
import os


# In[1]:


def airways(row):  
    if row['InvVentET'] == True or         row['InvVentTT'] == True or         row['Niv'] == True or         row['AvsJet'] == True or         row['AvsOsc'] == True or         row['AsthmaIVBeph'] == True or         row['Naso'] == True or         row['Trach'] == True or         row['OxTherapy'] == True or         row['Apnoea'] == True or         row['ObsAir'] == True:
        return 'Yes'
    return 'No'


# In[2]:


def cv(row):  
    if row['ArtLine'] == True or         row['ExtPace'] == True or         row['CvpMon'] == True or         row['InfInotrope'] == True or         row['Bolus'] == True or         row['Cpr'] == True or         row['Ecmo'] == True or         row['Vad'] == True or         row['AbPump'] == True or         row['ArrhythmiaAATherapy'] == True:
        return 'Yes'
    return 'No'


# In[3]:


def renal(row):  
    if row['PeriDia'] == True or         row['HaemoDia'] == True or         row['HaemoFilt'] == True or         row['PlasmaFilt'] == True or         row['PlasmaExch'] == True:
        return 'Yes'
    return 'No'


# In[4]:


def neuro(row):  
    if row['IcpMon'] == True or         row['IntCathEvd'] == True or         row['StatusEpilepticusAEDrugs'] == True or         row['LowGCS'] == True:
        return 'Yes'
    return 'No'


# In[5]:


def analgesic(row):  
    if row['EpiduralCatheter'] == True or         row['ContIVSedative'] == True:
        return 'Yes'
    return 'No'


# In[7]:


def metabolic(row):  
    if row['Dka'] == True:
        return 'Yes'
    return 'No'

def exclude_activity(df, activity):
    total = 0
    for event in activity:
        total += (df.shape[0]-df.loc[df.EventID != event].shape[0])
        df = df.loc[df.EventID != event]
    
    print(f'Number of records with unwanted events:\n{total}')
    
    return df


# In[9]:


def transform(df, activity, id_picanet,id_redcap):
    df['airways'] = df.apply(lambda row: airways(row), axis=1)
    df['cv'] = df.apply(lambda row: cv(row), axis=1)
    df['renal'] = df.apply(lambda row: renal(row), axis=1)
    df['neuro'] = df.apply(lambda row: neuro(row), axis=1)
    df['analgesic'] = df.apply(lambda row: analgesic(row), axis=1)
    df['metabolic'] = df.apply(lambda row: metabolic(row), axis=1)
    
    df = preprocess_picanet.clean_ids(df,id_picanet,id_redcap)
    
    df = exclude_activity(df, activity)
    
    # Find and save records without participant_ids
    print(f'Number of records with no ids:\n{df[df.index.isnull()].shape[0]}')
    df[df.index.isnull()].to_csv(os.path.join(preprocess_picanet.generating_data_files('Deletes'), 'no_ids_activity.csv'))
    
    df = df[~df.index.isnull()]
    
    df.to_csv(os.path.join(preprocess_picanet.generating_data_files('Data'), 'picanet_activity.csv'))    
    
    
    return df

def activity_summary(df, activity):
    for act in activity:
        
        # Count of presence of activity
        df_yes = (df.groupby(['EventID'])
                 .apply(lambda x: (x[act]== 'Yes').sum())
                 .reset_index(name='countYes'))
        df_yes = df_yes.set_index('EventID')

        # Count of absence of activity
        df_no = df.groupby(['EventID']).apply(lambda x: (x[act]== 'No').sum()).reset_index(name='countNo')
        df_no = df_no.set_index('EventID')

        # Merge dataframes
        df_merge = df_yes.merge(df_no, left_index=True, right_index=True)
        df_merge['Percentage'] = df_merge.apply(lambda x: x['countYes']/(x['countYes']+x['countNo']), axis=1)

        print('Summary stats of', act,'activity')
        print(df_merge.describe())
        
        print('Count of participants that had', act,'activity')
        print(len(df_merge.loc[df_merge['countYes'] == 0]))
    

