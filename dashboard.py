import streamlit as st 
from Environment import env_data
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import plotly.express as px 


# Set a custom CSS for background
st.markdown(
    """
    <style>
    .main{
        background-color: #FFFFFF;
    }
    .sidebar {
        background-color: #FF8C00;
    }
    .sidebar-content{
        color: #3CB371;
    }
    h1{
        color: #BA55D3;
    }
    h2, h3{
        color: #36454F;
    }
    p{
        color: black;
    }
    .st-bx{
        background-color: #FF8C00;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Singapore's Environmental Data Dashboard")

# Sidebar for selecting the category
st.sidebar.header("User Input Features")
category = st.sidebar.selectbox("Select Category to Visualize", env_data.columns)

# Additional Sidebar Widgets
year_range = st.sidebar.slider('Select Year Range', min_value=int(env_data['Year'].min()), 
                               max_value=int(env_data['Year'].max()), 
                               value=(int(env_data['Year'].min()), int(env_data['Year'].max())))

# Filter data based on user selection
filtered_data = env_data[(env_data['Year'] >= year_range[0]) & (env_data['Year'] <= year_range[1])]

# Customize color palette
sns.set_palette("coolwarm")

# Main Plot (Matplotlib/Seaborn)
st.subheader(f"{category} over Years")
plt.figure(figsize=(10, 6))
sns.lineplot(x='Year', y=category, data=filtered_data, marker='o', linewidth=2.5, color='#BA55D3')
plt.title(f'{category} Over Time', fontsize=16, color='#1e212b')
plt.xlabel('Year', fontsize=14, color='#1e212b')
plt.ylabel(category, fontsize=14, color='#1e212b')
plt.grid(True)
st.pyplot(plt)

# Optional: Interactive Plot using Plotly
if st.sidebar.checkbox("Show Plotly Interactive Plot"):
    st.subheader(f"Interactive Plot of {category}")
    fig = px.line(filtered_data, x='Year', y=category,title=f'{category} Over Time',
                  labels={'Year': 'Year', category: category},
                  template="plotly_dark", color_discrete_sequence=['#BA55D3'])
    fig.update_traces(line=dict(width=3))
    st.plotly_chart(fig)

# Summary statistics
st.write(f"Summary Statistics for {category}")
st.dataframe(filtered_data.describe())

# Scatter plot
if  st.sidebar.checkbox("Show Scatter Plot"):
    st.subheader(f"Scatter Plot of {category} vs Year")
    fig = px.scatter(filtered_data, x='Year', y=category, color=category, 
                     title=f'Scatter Plot of {category} vs Year',
                         template="plotly_white")
    st.plotly_chart(fig)










# Load your cleaned dataset
# Assuming df_filled is already prepared in your existing code
# Replace the following line with the actual import or dataset loading process


# # Streamlit app title
# st.title("Environmental Data Dashboard")

# # Sidebar for selecting the category (a column in your dataset)
# st.sidebar.header("Select Category")
# category = st.sidebar.selectbox("Choose a category to visualize", env_data.columns)

# # Filtering the dataset based on the selected category
# filtered_data = env_data[['Year', category]]  # Assuming 'Year' is a column for x-axis

# # Plotting the selected category against 'Year'
# st.subheader(f"Plot of {category} over Years")

# # Matplotlib/Seaborn plot
# plt.figure(figsize=(10, 6))
# sns.lineplot(x='Year', y=category, data=filtered_data, marker='o', color='hotpink')
# plt.title(f'{category} over Years')
# plt.xlabel('Year')
# plt.ylabel(category)
# plt.grid(True)
# st.pyplot(plt)

# # Additional Plots (Optional) - Customize based on your needs
# if category == 'specific_column_name':
#     st.subheader(f"Additional Plot for {category}")
#     # Add other visualizations or analyses here
#     # Example: sns.scatterplot or any other type of plot
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x='Year', y=category, data=filtered_data)
#     plt.title(f'Scatter plot of {category} over Years')
#     plt.xlabel('Year')
#     plt.ylabel(category)
#     plt.grid(True)
#     st.pyplot(plt)

# # To run the Streamlit app, save this script and run `streamlit run your_script_name.py` in your terminal
