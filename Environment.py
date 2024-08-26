#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymysql
import numpy as np


# In[ ]:


sns.set_style('whitegrid')


# In[ ]:


df = pd.read_csv('global-data-on-sustainable-energy.csv')
df


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


df.info()


# In[ ]:


df



# In[ ]:


df_filled = df.fillna({'Financial flows to developing countries (US $)': df['Financial flows to developing countries (US $)'].mean(),
                            'Renewables (% equivalent primary energy)': 0,
                            'Value_co2_emissions_kt_by_country': df['Value_co2_emissions_kt_by_country'].median(),
                            'Renewable energy share in the total final energy consumption (%)': df['Renewable energy share in the total final energy consumption (%)'].mean(),
                            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].mean()})
df_filled


# In[ ]:


df_sing = df[df['Entity'] == 'Singapore']
df_sing


# In[ ]:


df_sing.info


# In[ ]:


df_sing_filled = df_sing.fillna({'Financial flows to developing countries (US $)': df['Financial flows to developing countries (US $)'].mean(),
                            'Renewables (% equivalent primary energy)': 0,
                            'Value_co2_emissions_kt_by_country': df['Value_co2_emissions_kt_by_country'].median(),
                            'Renewable energy share in the total final energy consumption (%)': df['Renewable energy share in the total final energy consumption (%)'].mean(),
                            'Energy intensity level of primary energy (MJ/$2017 PPP GDP)': df['Energy intensity level of primary energy (MJ/$2017 PPP GDP)'].mean()})
df_sing_filled


# In[ ]:


df_sing.reset_index()


# In[ ]:


df_waste_1 = pd.read_csv('2003_2017_waste.csv')
df_waste_1


# In[ ]:


df_waste_2 = pd.read_csv('2018_2020_waste2.csv')
df_waste_2


# In[ ]:


df_waste_1['Year'] = df_waste_1['year']
df_waste_1.drop(columns='year', inplace=True)
df_waste_1


# In[ ]:


df_waste = pd.concat([df_waste_1, df_waste_2])
df_wastes = df_waste.reset_index()
df_wastes.drop('index', axis=1, inplace=True)
df_wastes


# In[ ]:


merged_df = pd.merge(df_sing_filled, df_wastes, on='Year')
# merged_df.to_csv('environment.csv', index=False)
env_data = merged_df[merged_df['Year'] != 2020]
# env_data= env_data.drop_duplicates(subset=['Waste Type'])


# In[ ]:


env_data


# In[ ]:


# # Establish a connection to MySQL
# db = pymysql.connect(
#     host="localhost",        # Replace with your host
#     user="root",    # Replace with your MySQL username
#     password="saganash",# Replace with your MySQL password
#     database="Environment" # Replace with your MySQL database name
# )

# cursor = db.cursor()
# # cursor.execute("INSERT INTO Environment (Entity, Year, Access to electricity (% of population),
#     Access to clean fuels for cooking,
#         Renewable-electricity-generating-capacity-per-capita,
#         Financial flows to developing countries (US $),
#         Renewable energy share in the total final energy consumption (%),
#         Electricity from fossil fuels (TWh), Electricity from nuclear (TWh),
#         Electricity from renewables (TWh),
#         Low-carbon electricity (% electricity),
#         Primary energy consumption per capita (kWh/person),
#         Energy intensity level of primary energy (MJ/$2017 PPP GDP),
#         Value_co2_emissions_kt_by_country,
#         Renewables (% equivalent primary energy), gdp_growth,
#         gdp_per_capita, Density\n(P/Km2), Land Area(Km2), Latitude,
#         Longitude, Waste Type, Total Recycled ('000 tonnes),
#         Total Generated ('000 tonnes)) VALUES (%(Entity)s, %(Year, Access to electricity (% of population))s,
#         %(Access to clean fuels for cooking)s,
#         %(Renewable-electricity-generating-capacity-per-capita)s,
#         %(Financial flows to developing countries (US $))s,
#         %(Renewable energy share in the total final energy consumption (%))s,
#         %(Electricity from fossil fuels (TWh), Electricity from nuclear (TWh))s,
#         %(Electricity from renewables (TWh))s,
#         %(Low-carbon electricity (% electricity))s,
#         %(Primary energy consumption per capita (kWh/person))s,
#         %(Energy intensity level of primary energy (MJ/$2017 PPP GDP))s,
#         %(Value_co2_emissions_kt_by_country)s,
#         %(Renewables (% equivalent primary energy))s, %(gdp_growth)s,
#         %(gdp_per_capita)s, %(Density\n(P/Km2))s, %(Land Area(Km2))s, %(Latitude)s,
#         %(Longitude)s, %(Waste Type)s, %(Total Recycled ('000 tonnes))s,
#         %(Total Generated ('000 tonnes))s)")


# In[ ]:


# from sqlalchemy import create_engine

# # Create a SQLAlchemy engine for MySQL using PyMySQL
# engine = create_engine('mysql+pymysql://root:saganash@localhost/Environment')



# In[ ]:


merged_df.columns


# In[ ]:


# # Example SQL to create a table
# create_table_query = """
# CREATE TABLE IF NOT EXISTS Environment (
#     id INT AUTO_INCREMENT PRIMARY KEY,
#     column1 INT,
#     column2 VARCHAR(255)
# )
# """

# cursor.execute(create_table_query)
# db.commit()


# In[ ]:


# engine = create_engine("mysql+pymysql://root:saganash@localhost/Environment", pool_pre_ping=True)


# In[ ]:


# Save the DataFrame to a MySQL table
# merged_df.to_sql('Environment', con=engine, if_exists='append', index=False)


# In[ ]:


merged_df


# In[ ]:


carbon_footprint = env_data['Value_co2_emissions_kt_by_country']
carbon_footprint


# In[ ]:


# Bar plot on carbon emission
plt.figure(figsize=(16, 8))
sns.barplot(data=env_data, x='Year', y='Value_co2_emissions_kt_by_country', palette='viridis')
plt.title('Carbon Emission in Singapore')
plt.xlabel('Year')
plt.ylabel('Carbon Emission in KiloTons(kt)')
# plt.savefig('carbon_emissions.png')
plt.show()


# ##### Findings and Insights
# The Bar plot above represents Singapore's Carbon Emissions over time in Years
#  - There is an increase of carbon emissions from about 40000 kt in 2003 to under 50000 kt in 2019.
#  - Emissions are relatively stable from 2003-2009 with some fluctuations around 40000 kt.
#  - There is a gradual increase in carbon emissions from 2010-2019
#  - Overall, the upward trend highlights concerns on sustainability and the need for further emission reduction efforts. 
# 

# In[ ]:


plt.figure(figsize=(14, 8))
sns.barplot(data=env_data, x="Year",y="Total Recycled ('000 tonnes)", palette='viridis', ci=None)
plt.title('Recycled Watse in Singapore')
plt.xlabel("Year")
plt.ylabel("Total Recycled Waste ")
# plt.savefig("Recycled_Waste.png")
plt.show()


# #### Findings and Insights
# The bar plot above shows the total recycled waste with time in Year 
#  - The total recycled waste rises from 300000 in 2003 to above 600000 in 2017.
#  - The amount of total recycled waste gradually rises from 2003 to 2013.
#  - The amount of total recycled waste is relatively stable from 2013 to 2017 with minor fluctuations suggesting a plateau in recycling efficiency
#     

# In[ ]:


plt.figure(figsize=(20, 8))
sns.barplot(data=env_data, x="Year",y="Total Generated ('000 tonnes)", palette='viridis', ci=None)
plt.title('Generated Watse in Singapore')
plt.xlabel("Year")
plt.ylabel("Total Generated Waste ('00000 Tonnes)")
# plt.savefig('Generated Waste.png')
plt.show()


# #### Finding and Insight
# The Barplot above shows Generated Waste in Singapore over time in Years.
#  - There is rise in amount of generated waste from 600000T in 2003 to the peak of about 1000000T in 2013
#  - There is a gradual rise of generated waste from 2004 to 2013.
#  - The amount of generated waste is relatively stable from 2013-2017 having plateau trend suggesting an impact in waste reduction methods

# In[ ]:


plt.figure(figsize=(20, 8))
# Width of the bars
bar_width = 0.35  # Make the bars narrower so they fit side by side

# Unique years for the x-axis
years = env_data['Year'].unique()
r1 = np.arange(len(years))  # Only one entry per year
r2 = r1 + bar_width  # Offset for the second bar

# Create the bar chart
plt.bar(r1, env_data.groupby('Year')["Total Recycled ('000 tonnes)"].sum(), color='hotpink', alpha=0.6, width=bar_width, edgecolor='grey', label='Total Recycled')
plt.bar(r2, env_data.groupby('Year')["Total Generated ('000 tonnes)"].sum(), color='cyan', alpha=0.6, width=bar_width, edgecolor='grey', label='Total Generated')

# Adding labels
plt.xlabel('Year', fontweight='bold')
plt.ylabel('Values', fontweight='bold')
plt.xticks(r1 + bar_width / 2, years)
plt.xlabel('Year')
plt.ylabel("Waste metrics ('000 tonnes)")
plt.title('Waste Management over Time')
# Adding legend
plt.legend()
# plt.savefig('waste.png')
# Show the plot
plt.show()


# #### Findings and Insights
#  - Both Recycled and Generated waste had an upward trend from 2003-2017
#  - In all years, generated waste is generally higher than Recycled waste, widening  the gap over tie especially past 2010
#  - From 2014-2017 the recycling efforts appear to have stagnated showing slower growth compared to the waste generated.
#  - Companies need to focus on reducing waste production and scaling up effort on recycling waste to bridge the gap between waste recycled and generated

# In[ ]:


plt.figure(figsize=(14, 8))
plt.plot(env_data['Year'], env_data['Renewable energy share in the total final energy consumption (%)'], label='Renewable Energy', marker='x', color='hotpink')
plt.xlabel('Year')
plt.ylabel('Renewable energy share in the total final energy consumption (%)')
plt.title('Renewable Energy Share in Singapore')
# plt.savefig('Renewable Energy Share.png')
plt.show()


# #### Findings and Insights
#  The lineplot above represents the Renewable energy share in the total final Energy consumption over time.
#  - The reneable energy share in the total final energy consumption has show a significant upward trend from 2003-2019.
#  - The share slightly declined from 2003-2010 showing that there were some challenges faced durng this period.
#  - From 2010 onwards there has been a significant increase in renewable energy share.
#  - The data reflects Singapore's growing commitment to sustainable energy with substantial improvements made in the recent years.

# In[ ]:


plt.figure(figsize=(14, 6))
# plt.plot(env_data["Year"], env_data["Electricity from fossil fuels (TWh)"], label='Fossil Fuels', marker='x',color='purple')
plt.plot(env_data["Year"], env_data["Electricity from nuclear (TWh)"], label='Nuclear', marker='x', color='hotpink')
plt.plot(env_data["Year"], env_data["Electricity from renewables (TWh)"], label='Renewables', marker='v', color='green')
plt.xlabel("Year")
plt.ylabel("Electricity Sources (TWh)")
plt.title('Comparison of Electricity Sources over Time')
plt.legend()
# plt.savefig('Electricity sources over time.png')
plt.show()


# #### Findings and Insights 
#    - The contribution of Nuclear energy has remained constant over the period
#    - Renewable energy shows a sharp increase in its contribution starting around 2012.
#    - This growth accelerates significantly after 2014, reaching the highest level in 2019 with approximately 0.8 TWh.
#    - The data highlights a clear trend towards renewable energy as the dominant growth sector in electricity generation, while nuclear energy has remained largely static
# 

# In[ ]:


plt.figure(figsize=(14, 6))
plt.plot(env_data["Year"], env_data["Electricity from fossil fuels (TWh)"], label='Fossil Fuels', marker='x',color='purple')
# plt.plot(env_data["Year"], env_data["Electricity from nuclear (TWh)"], label='Nuclear', marker='x', color='hotpink')
# plt.plot(env_data["Year"], env_data["Electricity from renewables (TWh)"], label='Renewables', marker='v', color='green')
plt.xlabel("Year")
plt.ylabel("Electricity Sources (TWh)")
plt.title('Comparison of Electricity Sources over Time')
plt.legend()
# plt.savefig('Fossil fuels.png')
plt.show()


# #### Findings and Insights
# - The use of fossil fuels has seen a significant growth trend from 2002-2019
# - The peak was under 50TWh

# In[ ]:


# Assuming data is your original DataFrame
grouped_data = env_data.groupby(['Year', 'Waste Type']).agg({
    'Total Recycled (\'000 tonnes)': 'sum',
    'Total Generated (\'000 tonnes)': 'sum'
}).reset_index()

# Display the grouped data
grouped_data


# Assuming grouped_data is the DataFrame obtained from the grouping and aggregation step
pivoted_data = grouped_data.pivot_table(
    index='Year',               # Rows will be indexed by Year
    columns='Waste Type',       # Columns will be the different Waste Types
    values=['Total Recycled (\'000 tonnes)', 'Total Generated (\'000 tonnes)'],  # Values to pivot
    aggfunc='sum'               # Aggregation function, in this case, sum
)

# Flatten the MultiIndex in columns
pivoted_data.columns = ['_'.join(col).strip() for col in pivoted_data.columns.values]

# Reset index to make 'Year' a column again
pivoted_data = pivoted_data.reset_index()

# Display the pivoted data
pivoted_data_filled = pivoted_data.fillna(0)
pivoted_data_filled


# In[ ]:


# Plotting Generated Waste
plt.figure(figsize=(14, 7))

# Plot lines for each waste type (generated)
for waste_type in pivoted_data_filled.columns[2::2]:  # Assuming every second column is a generated waste column
    plt.plot(pivoted_data_filled['Year'], pivoted_data_filled[waste_type], marker='x', label=waste_type)

plt.title('Total Generated Waste by Waste Type Over Years')
plt.xlabel('Year')
plt.ylabel('Total Generated (\'000 tonnes)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.show()

# Plotting Recycled Waste
plt.figure(figsize=(14, 7))

# Plot lines for each waste type (recycled)
for waste_type in pivoted_data_filled.columns[1::2]:  # Assuming every second column is a recycled waste column
    plt.plot(pivoted_data_filled['Year'], pivoted_data_filled[waste_type], marker='d', label=waste_type)

plt.title('Total Recycled Waste by Waste Type Over Years')
plt.xlabel('Year')
plt.ylabel('Total Recycled (\'000 tonnes)')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
# plt.savefig('waste type.png')
plt.show()



# #### Findings and Insights
# - The total recycled waste increased significantly from 2004 to 2017, peaking at around 7 million tonnes in 2017, but saw a sharp decline in 2018. Most individual waste types show little variation, contributing a small portion to the overall trend.

# In[ ]:


plt.figure(figsize=(14, 8))
plt.plot(env_data['Year'], env_data['Primary energy consumption per capita (kWh/person)'], label='Energy Consumption', marker='v', color='cyan')
plt.title('Primary Energy Consumption per Capita over Time in Singapore')
plt.xlabel('Year')
plt.ylabel("Primary Energy consumption per capita (kWh/person)")
plt.legend()
# plt.savefig('energy consumption.png')
plt.show()


# #### Findings and Insights
# - Energy consumption per capita in Singapore steadily increased from 2004 to 2013, reaching a peak around 2014, after which it plateaued and slightly decreased by 2018.

# In[ ]:


plt.figure(figsize = (12, 8))
sns.lineplot(x="Value_co2_emissions_kt_by_country", y="gdp_growth", data=env_data, color='hotpink', marker='o', label='Carbon Footprint', alpha= 0.6)
plt.ylabel("GDP Growth")
plt.xlabel("Carbon Emissions in KiloTonnes(kt)")
plt.title("GDP Growth against Carbon Emissions")
# plt.savefig('gdp against carbon emission.png')
plt.show()


# #### Findings and Insights
# - There is an inverse relationship between GDP growth and carbon emissions. As carbon emissions increased in the early 2010s, GDP growth fluctuated, peaking significantly at one point, but began declining in response to higher carbon emissions.

# In[ ]:


env_data.columns


# In[ ]:


numeric_df = env_data.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr(method='pearson', min_periods=1)
corr_matrix_cleaned = correlation_matrix.dropna(axis=1, how='all').dropna(axis=0, how='all')
gdp_corr = corr_matrix_cleaned['gdp_growth']
# zero_variance_columns = correlation_matrix.columns[numeric_df.var() == 0]
# df_cleaned = correlation_matrix.drop(zero_variance_columns, axis=1)
df_cleaned = gdp_corr.drop('gdp_growth')
sorted_corr = df_cleaned.sort_values(ascending=False)
most_positively_correlated_column = sorted_corr.index[0]
most_positively_correlated_value = sorted_corr.iloc[0]

print(f"Most Positively Correlated column: {most_positively_correlated_column}")
print(f"Most Positively Correlated value: {most_positively_correlated_value:.3f}")


# In[ ]:


sorted_corr = df_cleaned.sort_values(ascending=True)
most_negatively_correlated_column = sorted_corr.index[0]
most_negatively_correlated_value = sorted_corr.iloc[0]

print(f"Most Negatively Correlated column: {most_negatively_correlated_column}")
print(f"Most Negatively Correlated value: {most_negatively_correlated_value:.3f}")


# In[ ]:


corr_matrix_cleaned


# In[ ]:





# In[ ]:


plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix_cleaned, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
plt.title('Correlation heatmap of numeric variables')
plt.savefig('correlation heatmap.png')
plt.show()


# In[ ]:


# Define a function to format the tick labels
def format_ticks(value, tick_number):
    return f'{value:.2f}'  # Adjust the number of decimal places as needed
plt.figure(figsize=(10, 6))
sns.boxenplot(x="Energy intensity level of primary energy (MJ/$2017 PPP GDP)", y='gdp_growth', data=env_data, color='gold')
sns.stripplot(x='Energy intensity level of primary energy (MJ/$2017 PPP GDP)', y='gdp_growth', data=env_data, color='gold', alpha=0.5)
# Get current x-axis tick values
ticks = plt.gca().get_xticks()

# Format tick labels to 2 decimal places
formatted_ticks = [f'{tick:.2f}' for tick in ticks]

# Set new tick labels
plt.gca().set_xticklabels(formatted_ticks, rotation=45, ha='left')
plt.title("Correlation of GDP Growth with Energy Intensity level")
plt.ylabel('GDP Growth')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 8))
sns.boxenplot(data=env_data, x='Renewable energy share in the total final energy consumption (%)', y='gdp_growth', color='gold')
plt.title('Correlation of GDP Growth with Renewable Energy share in total')
plt.ylabel('GDP Growth')
plt.savefig('corr eng intensity.png')
plt.show()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Environment.ipynb')


# In[ ]:




