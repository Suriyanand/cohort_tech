#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv(r"D:\Development\vs.code\Data Science\data Analaytis\study problems\datasets\cohorts.csv")
df


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.dropna()


# In[6]:


df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# In[7]:


df.dtypes


# In[8]:


descriptive_stats = df.describe()
print(descriptive_stats)


# In[9]:


import plotly.graph_objects as go
import plotly.express as px

import plotly.io as pio
pio.templates.default = "plotly_white"

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df['Date'], y=df['New users'], mode='lines+markers', name='New Users'))
fig1.add_trace(go.Scatter(x=df['Date'], y=df['Returning users'], mode='lines+markers', name='Returning Users'))
fig1.update_layout(title='Trend of New and Returning Users Over Time',
                  xaxis_title='Date',
                  yaxis_title='Number of Users')
fig.show()


# In[ ]:


fig2 = px.line(data_frame = df, x = "Date", y = ["Duration Day 1","Duration Day 7"], markers = True,labels = {"distribution":"values"})
fig2.update_layout(title = "identify trends and duration between 1 to 7 days", xaxis_title = "Date" , yaxis_title = "Duration")
fig2.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


corr_matrix = df.corr()


# In[12]:


plt.figure(figsize = (12,6))
sns.heatmap(corr_matrix, annot = True, cmap = "coolwarm", fmt = '.2f')
plt.title("correlation of matrics varible")
plt.show()


# In[13]:


df["Week"] = df["Date"].dt.isocalendar().week


# In[14]:


avg_weekly_data = df.groupby('Week').agg({
    "New users":'mean',
    "Returning users":'mean',
    "Duration Day 1":'mean',
    "Duration Day 7":'mean'
}).reset_index()
print(avg_weekly_data)


# In[15]:


fig3 = px.line(data_frame = avg_weekly_data, x = "Week", y = ["New users","Returning users"], markers = True)
fig3.update_layout(title = "avg between the new users and returning users",
                 xaxis_title = "Number of users",yaxis_title = "Weak of the year")
fig3.show()


# In[16]:


fig4 = px.line(data_frame = avg_weekly_data, x = "Week", y = ["Duration Day 1","Duration Day 7"], markers = True)
fig4.update_layout(title = "avg between the Duration day 1 and Duration day 7",
                 xaxis_title = "Number of users",yaxis_title = "Weak of the year")
fig4.show()


# In[17]:


cohort_matrics = avg_weekly_data.set_index('Week')
plt.figure(figsize=(12,6))
sns.heatmap(cohort_matrics , annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Cohort Matrix of Weekly Averages')
plt.ylabel('Week of the Year')
plt.show()


# In[ ]:





# In[ ]:





# In[18]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define dropdown options
dropdown_options = [
    {'label': 'Trend of New and Returning Users Over Time', 'value': 'plot1'},
    {'label': 'identify trends and duration between 1 to 7 days', 'value': 'plot2'},
    {'label': 'avg between the new users and returning users', 'value': 'plot3'},
    {'label': 'avg between the Duration day 1 and Duration day 7', 'value': 'plot4'},

    
]

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Dashboard'),

    # Dropdown menu
    dcc.Dropdown(
        id='plot-dropdown',
        options=dropdown_options,
        value='plot1'  # Default value
    ),

    # Div to hold the selected plot
    html.Div(id='plot-container'),

])

# Callback to update the selected plot based on dropdown value
@app.callback(
    Output('plot-container', 'children'),
    [Input('plot-dropdown', 'value')]
)
def update_plot(selected_plot):
    if selected_plot == 'plot1':
        return dcc.Graph(id='graph1', figure=fig1) 
    # Plot 1
    elif selected_plot == 'plot2':
        return dcc.Graph(id='graph2', figure=fig2)
    
    elif selected_plot == 'plot3':
        return dcc.Graph(id='graph3', figure=fig3)
    elif selected_plot == 'plot4':
        return dcc.Graph(id='graph4', figure=fig4)
    # Plot 2
    # Add more elif statements for additional plots

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8056)


# In[19]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

# Calculate cohort matrix and correlation matrix here
# Assuming you already have calculated avg_weekly_data, cohort_matrics, and corr_matrix

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1('Heatmap Dashboard'),
    
    # First heatmap
    html.Div([
        dcc.Graph(id='heatmap1', figure=px.imshow(cohort_matrics, labels=dict(color="Value"))),
        html.H4("Cohort Matrix of Weekly Averages")
    ]),
    
    # Second heatmap
    html.Div([
        dcc.Graph(id='heatmap2', figure=px.imshow(corr_matrix, labels=dict(color="Value"))),
        html.H4("Correlation of Metrics Variables")
    ]),
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8058)


# In[ ]:




