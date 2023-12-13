import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.display.max_rows = 150
pd.options.display.max_columns = 150


def dataframe_retrieve(filename):
    '''
    retrieving 2 dataframes - yearwise and countrywise.
    '''
    df_year_feat = pd.read_excel(filename,engine="openpyxl")
    df_country_feat = pd.melt(df_year_feat, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                    var_name='Year', value_name='Value')
    df_country_feat = df_country_feat.pivot_table(index=['Year', 'Country Code', 'Indicator Name', 'Indicator Code'], columns='Country Name', values='Value').reset_index()
    df_country_feat = df_country_feat.drop_duplicates().reset_index()
    return df_year_feat,df_country_feat


df_year_feat,df_country_feat = dataframe_retrieve('world_bank_climate.xlsx')



def yearly_data(data,p,q,r):
    data_to_retrieve = data.copy()
    need_year=[i for i in range(p,q,r)]
    col_req=['Country Name','Indicator Name']
    col_req.extend(need_year)
    data_to_retrieve =  data_to_retrieve[col_req]
    data_to_retrieve = data_to_retrieve.dropna(axis=0, how="any") 
    return data_to_retrieve


sample_df = yearly_data(df_year_feat,1995,2020,3)

countries_to_analyze = sample_df['Country Name'].value_counts().index.tolist()[37:50]



def feature_value_selected(data,column,values):
    df_rand= data.copy()
    df_rand= df_rand[df_rand[column].isin(values)].reset_index(drop=True)
    return df_rand


sample_df_country  = feature_value_selected(sample_df,'Country Name',countries_to_analyze)


country_dict = dict()
for i in range(sample_df_country.shape[0]):
    if sample_df_country['Country Name'][i] not in country_dict.keys():
        country_dict[sample_df_country['Country Name'][i]]=[sample_df_country['Indicator Name'][i]]
    else:
        country_dict[sample_df_country['Country Name'][i]].append(sample_df_country['Indicator Name'][i])
    

for k,v in country_dict.items():
    country_dict[k] = set(v)


inter = country_dict['Guinea']
for v in country_dict.values():
    inter = inter.intersection(v)



print(ample_df_country.describe())


df_arable= feature_value_selected(sample_df_country,'Indicator Name',['Arable land (% of land area)'])


print(df_arable.describe())

df_arable  = feature_value_selected(df_arable,'Country Name',countries_to_analyze)


def bar_graph_country(data,feature_to_verify):
    df_filt = data.copy()
    df_filt.set_index('Country Name', inplace=True)
    num_col = df_filt.columns[df_filt.dtypes == 'float64']
    df_filt = df_filt[num_col]
    plt.figure(figsize=(50, 50))
    df_filt.plot(kind='bar')
    plt.title(feature_to_verify)
    plt.xlabel('Country Name')    
    plt.legend(title='Year', bbox_to_anchor=(1.10, 1), loc='upper left')
    plt.show()


bar_graph_country(df_arable,'Arable land (% of land area)')


df_no2 = feature_value_selected(sample_df_country,'Indicator Name',['Nitrous oxide emissions (thousand metric tons of CO2 equivalent)'])

print(df_no2.describe())


bar_graph_country(df_no2,'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)')


df_japan= feature_value_selected(sample_df_country,'Country Name',['Japan'])

def indicator_filtered_data(data):
    df_filtered=data.copy()
    # Melt the DataFrame
    df_filtered = df_filtered.melt(id_vars='Indicator Name', var_name='Year', value_name='Value')

    # Pivot the DataFrame
    df_filtered = df_filtered.pivot(index='Year', columns='Indicator Name', values='Value')

    # Reset index
    df_filtered.reset_index(inplace=True)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')
    del df_filtered['Year']
    df_filtered = df_filtered.rename_axis(None, axis=1)
    return df_filtered



df_japan= indicator_filtered_data(df_japan)


features_filt = ['Forest area (% of land area)',
 'Cereal yield (kg per hectare)',
                     'Arable land (% of land area)',
 'Nitrous oxide emissions (thousand metric tons of CO2 equivalent)',
 'Urban population growth (annual %)',
                'Renewable energy consumption (% of total final energy consumption)']



df_japan = df_japan[features_filt]

print(df_japan.corr())



sns.heatmap(df_japan.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')



df_cereal= feature_value_selected(sample_df_country,'Indicator Name',['Cereal yield (kg per hectare)'])


print(df_cereal.describe())

df_urban= feature_value_selected(sample_df_country,'Indicator Name',['Urban population growth (annual %)'])



def graph_for_year(data,indicator_variable):
    df_ind = data.copy()
    df_ind.set_index('Country Name', inplace=True)
    num_check = df_ind.columns[df_ind.dtypes == 'float64']
    df_ind = df_ind[num_check]

    plt.figure(figsize=(12, 6))
    for count in df_ind.index:
        plt.plot(df_ind.columns, df_ind.loc[count], label=count, linestyle='dashed', marker='o')

    plt.title(indicator_variable)
    plt.xlabel('Year')
    plt.legend(title='Country', bbox_to_anchor=(1.15, 1), loc='upper left')

    plt.show()



graph_for_year(df_cereal,'Cereal yield (kg per hectare)')


df_ren= feature_value_selected(sample_df_country,'Indicator Name',['Renewable energy consumption (% of total final energy consumption)'])

print(df_ren.describe())


graph_for_year(df_ren,'Renewable energy consumption (% of total final energy consumption)')



df_uruguay = feature_value_selected(sample_df_country,'Country Name',['Uruguay'])
df_uruguay = indicator_filtered_data(df_uruguay)
df_uruguay = df_uruguay[features_filt]
sns.heatmap(df_uruguay.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')

df_parag= feature_value_selected(sample_df_country,'Country Name',['Paraguay'])
df_parag = indicator_filtered_data(df_parag)
df_parag = df_parag[features_filt]
plt.figure()
sns.heatmap(df_parag.corr(), annot=True, cmap='YlGnBu', linewidths=.5, fmt='.3g')



