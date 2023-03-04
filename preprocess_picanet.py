



import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import glob
import re
from datetime import datetime
from dateutil import relativedelta
import os


# In[2]:


def clean_ids(df,col1, col2):
    """ Format ID to represent participant_id-> 4 characters and 4 numbers 
    
    OR family_id-> 3 numbers-OCE+4 numbers.
    
    """
    
    df[col1] = df[col1].str.replace(" ","")
    df[col1] = df[col1].str.extract(r'([a-zA-z]{4}-[0-9]{4}|0[0-9]{2}-[Oo0][Cc][Ee][0-9]{4})')

    #ID column is renamed so that it can be linked with the redcap data
    df = df.rename(columns={col1:col2}).set_index(col2)
    
    df.index = df.index.str.upper()
    
    return df
    


# In[3]:


def feature_selection(df,feat_list):
    """ Feature selection """
    df = df[feat_list]
    
    return df


# In[4]:


def age_formating(df, date_feats, adm_age=True, adm_days=True):
    """ Format dates to datetime format.
    AND Calculate the admission age of participant
    AND Calculate days admitted
    
    """
    for i in range(len(date_feats)):
        df[date_feats[i]]= pd.to_datetime(df[date_feats[i]], dayfirst=True)
    
    #calculate Admission Age
    if (adm_age == True):
        df['AdAgeYears'] = (df['AdDate']-df['Dob']).astype('timedelta64[Y]')
        df['AdAgeMonths'] = (df['AdDate']-df['Dob']).astype('timedelta64[M]')
        df['AdAge'] = df['AdAgeMonths'].apply(lambda x: divmod(x,12))

    #calculate number of date admitted
    if (adm_days == True):
        df['AdDays'] = (df['UnitDisDate']-df['AdDate']).astype('timedelta64[D]')
    
    return df


# In[5]:


def generating_results(x):
    import os
    # creating results folder, if it doesnt exist
    current_directory = os.getcwd()
    result_directory = os.path.join(current_directory, r'Results')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    specific_directory = os.path.join(result_directory, x)
    if not os.path.exists(specific_directory):
        os.makedirs(specific_directory)
    return specific_directory


# In[6]:


def generating_data_files(x):
    import os
    # creating results folder, if it doesnt exist
    current_directory = os.getcwd()
    result_directory = os.path.join(current_directory, r'Preprocessed_data')
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    specific_directory = os.path.join(result_directory, x)
    if not os.path.exists(specific_directory):
        os.makedirs(specific_directory)
    return specific_directory


# In[7]:


def remove_duplicates(df):
    df = df.reset_index(level=0)
    # Find and save duplicate records
    print(f'Number of duplicate records:\n{df.loc[df.duplicated(keep=False)].shape[0]}')
    df.loc[df.duplicated(keep=False)].set_index(df.columns[0]).sort_index().to_csv(os.path.join(generating_data_files('Deletes'), 'duplicate_records.csv'))
    
    df = df.loc[~df.duplicated()].set_index(df.columns[0])
    
    # Find and save records with the same participant id but different admission dates, reasons for admission, admission type etc
    print(f'Number of records with same participant id but different admission dates:\n{df.loc[df.index.duplicated(keep=False)].shape[0]}')
    df.loc[df.index.duplicated(keep=False)].sort_index().to_csv(os.path.join(generating_data_files('Deletes'), 'multiple_id_records.csv'))

    
    # Find and save records without participant_ids
    print(f'Number of records with no ids:\n{df[df.index.isnull()].shape[0]}')
    df[df.index.isnull()].to_csv(os.path.join(generating_data_files('Deletes'), 'no_ids.csv'))
    
    df = df[~df.index.isnull()]
    
    return df


# In[ ]:


def preprocess(dfn, col1, col2, feat_list, date_feats, adm_age, adm_days):
    df = clean_ids(dfn,col1, col2)
    df = feature_selection(df, feat_list)
    df = age_formating(df,date_feats, adm_age, adm_days)
    df = remove_duplicates(df)
    
    df.to_csv(os.path.join(generating_data_files('Data'), 'picanet_admissin.csv'))
    
    return df
    
    
    


# In[8]:


def plot_summary_stats(df):
    
    df_distinct = df.loc[~df.index.duplicated()]
    
    # Sex 
    print('Number of Participants by Sex:\n',df_distinct.Sex.value_counts())
    print('Percentage distribution of participants by Sex:\n',df_distinct.Sex.value_counts(normalize=True).mul(100).
          rename_axis('Sex').reset_index(name='percentage').round(2))
    df_distinct['Sex'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Gender', title='Gender Distribution across PICAnet Data', figsize=(9,9))
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'sex.png'), dpi=300)
    plt.show()
    
    df_distinct.boxplot(column = ['AdAgeYears'], by= 'Sex', figsize=(9,9), grid= False )
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'sex_age.png'), dpi=300)
    plt.show()
    
    # Ethnicity
    print('Number of Participants by Ethnicity\n',df_distinct.EthnicDescription.value_counts())
    print('Percentage distribution of participants by Ethnicity:\n',df_distinct['EthnicDescription'].value_counts(normalize=True).mul(100).
          rename_axis('EthnicDescription').reset_index(name='percentage').round(2))

    df_distinct['EthnicDescription'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Ethnicity', title='Ethnicity across PICAnet Data', figsize=(11,5), color='red')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Ethnicity.png'), dpi=300)
    plt.show()
    
    
    # Admission type
    print('Number of Participants by Admission Type\n',df.AdTypeDescription.value_counts())
    print('Percentage distribution of participants by Admission Type:\n',df['AdTypeDescription'].value_counts(normalize=True).mul(100).
          rename_axis('AdTypeDescription').reset_index(name='percentage').round(2))

    df['AdTypeDescription'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Type of Admission', title='Admission Type for PICAnet Data', figsize=(9,5), color='green')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Admission_Type.png'), dpi=300)
    plt.show()
    
    # Admission Age
    print('Number of Participants by Admission Age\n',df.AdAgeYears.value_counts())
    print('Percentage distribution of participants by Admission Age:\n',df['AdAgeYears'].value_counts(normalize=True).mul(100).
          rename_axis('AdAgeYears').reset_index(name='percentage').round(2))
    
    
    df['AdAgeYears'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Age (Years)', title='Age during Admission for PICAnet Data', figsize=(9,5), color='blue')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Admission_Age.png'), dpi=300)
    plt.show()
    
    
    # Reason for admission 
    print('Number of Participants by Reason for Admission\n',df.PrimReasonDescription.value_counts())
    print('Percentage distribution of participants by Reason for Admission:\n',df['PrimReasonDescription'].value_counts(normalize=True).mul(100).
          rename_axis('PrimReasonDescription').reset_index(name='percentage').round(2))

    df['PrimReasonDescription'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Reason', title='Reason for PICU admission for PICAnet Data', figsize=(9,5), color = 'green')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Admission_Reason.png'), dpi=300)
    plt.show()   
    
    # Source of admission
    print('Number of Participants by Source of Admission\n',df.SourceAdDescription.value_counts())
    print('Percentage distribution of participants by Source of Admission:\n',df['SourceAdDescription'].value_counts(normalize=True).mul(100).
          rename_axis('SourceAdDescription').reset_index(name='percentage').round(2))
    
    df['SourceAdDescription'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='Source', title='Source of Admission for PICAnet Data', figsize=(11,5), color='red')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Admission_Source.png'), dpi=300)
    plt.show() 
    
    # Retrieval description
    print('Number of Participants by Retrieval\n',df.RetrievalDescription.value_counts())
    print('Percentage distribution of participants by Retrieval:\n',df['RetrievalDescription'].value_counts(normalize=True).mul(100).
          rename_axis('RetrievalDescription').reset_index(name='percentage').round(2))

    df['RetrievalDescription'].value_counts().plot(kind='bar', ylabel='Retrieval', xlabel='Source', title='Retrieval for PICAnet Data', figsize=(11,5))
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'RetrievalDescription.png'), dpi=300)
    plt.show() 
    
    # Type of transport team
    print('Number of Participants by Transport Org\n',df.ATransportOrgTypeDescription.value_counts())
    print('Percentage distribution of participants by Transport Org:\n',df['ATransportOrgTypeDescription'].value_counts(normalize=True).mul(100).
          rename_axis('ATransportOrgTypeDescription').reset_index(name='percentage').round(2))

    df['ATransportOrgTypeDescription'].value_counts().plot(kind='bar', ylabel='Frequency', xlabel='transport type', title='Type of Transport Team for PICAnet Data', figsize=(11,5), color='red')
    plt.savefig(os.path.join(generating_results('Picanet Summary stats'), 'Transport_type.png'), dpi=300)
    plt.show()
    
    
    #view summary statistics
    print('Summary statistics of numerical features\n')
    print(df.describe())
      

