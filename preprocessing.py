import pandas as pd
import numpy as np
import pickle

# IRT DATA
#path = '/Volumes/GoogleDrive/My Drive/Current Work/Current Projects/Regulatory Barrier Measurement/Data/'
data0610 = pd.read_csv('data/IRT data 0610 backup.csv')
data1115 =  pd.read_csv('data/IRT data 1115 backup.csv')
data = pd.concat([data0610, data1115], axis = 0)

# Country Name Data
#datapath = '/Users/hge/Google Drive/Current Work/Current Projects/Regulatory Barrier Measurement/Data/'
cname = pd.read_csv('data/country name list.csv')['0']


# Exclude countries with more than 95% values as 3
# column names of reporting entries
report_col = [str(i) for i in range(253)]
valid_col = (data.loc[:,report_col] == 3).mean(axis = 0) < 0.95
exclude_col = (data== 3).mean(axis = 0) >=0.95
exclude_col_name = exclude_col[exclude_col == True].index
# Check cnames
cname[valid_col.to_numpy]
# Cut data
data_cut = data.drop(exclude_col_name, axis = 1)



# Only leave firms that exist throughout this period
panel_cik = data_cut.loc[data_cut['year'] == 2006, 'cik'].drop_duplicates()
for t in data_cut['year'].unique():
    one_year = data_cut.loc[data_cut['year'] == t]
    one_year_cik = one_year['cik']
    panel_cik = panel_cik[panel_cik.isin(one_year_cik)]

# Most informative firms
# Choose top 500 firms with least 3s
most_informative_firms_rank = (data_cut == 3).mean(axis = 1).sort_values().index
informative_cik = data_cut['cik'].loc[most_informative_firms_rank]

informative_firm_id = panel_cik[panel_cik.isin(set(informative_cik[1:3500]))]
len(informative_firm_id)

# Cut firms
data_cut_firm = data_cut.loc[data_cut['cik'].isin(informative_firm_id)]


# Fix the first country as lowest barrier, last country as highest barrier
# the first country is Singapore
singapore_col = cname[cname == 'Singapore'].index[0]
# Switch the first column of cayman_col
temp = data_cut[str(singapore_col)]
data_cut.loc[:,str(singapore_col)] = data_cut.iloc[:,1] # the first column is index. the second col is the first country
data_cut.iloc[:,1] = temp   

# the last country is Russia
russia_col = cname[cname == 'Russia'].index[0]
temp = data_cut[str(russia_col)]
data_cut.loc[:,str(russia_col)] = data_cut.iloc[:,-3] # the last column is year. the penultimate col is cik
data_cut.iloc[:,-3] = temp  

# Swtich Cname
used_cname_list = cname[valid_col.to_numpy]
used_cname_list[10] = 'Singapore'
used_cname_list[singapore_col] = 'Argentina'
used_cname_list[241] = 'Russia'
used_cname_list[russia_col] = 'United Kingdom'


# Convert to 3D array: firm_country_year
num_firm = len(informative_firm_id)
num_country = sum(valid_col)
num_year = len(data_cut_firm['year'].unique())

# Order firms
firm_ord = informative_firm_id
firm_ord = firm_ord.reset_index(drop = True)


data_3d = np.full((num_firm, num_country, num_year), np.nan)
for t in data_cut['year'].unique():
    
    one_year = data_cut_firm.loc[data_cut_firm['year'] == t]
    firm_slice = one_year[one_year['cik'].isin(informative_firm_id)]
    # deal with duplicate
    firm_slice_unique = firm_slice[~firm_slice['cik'].duplicated()]
    
    # order according to cik
    firm_slice_unique.loc[:,'cik_cat'] = pd.Categorical(firm_slice_unique['cik'], categories=firm_ord, ordered=True)
    firm_slice_unique = firm_slice_unique.sort_values('cik_cat').reset_index(drop = True)
    
    if (firm_slice_unique['cik_cat'] != firm_ord).any():
        raise ValueError('Order of firms is wrong')
    
    # plug back
    data_3d[:,:,(t - 2006)] = firm_slice_unique.loc[:,valid_col.loc[valid_col == True].index]


# save
irt_data = {'data': data_3d, 'cname': used_cname_list}

with open('data/estimation_data.pickle', 'wb') as handle:
    pickle.dump(irt_data, handle, protocol=pickle.HIGHEST_PROTOCOL)



    
    
