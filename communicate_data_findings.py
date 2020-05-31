#!/usr/bin/env python
# coding: utf-8

# # Analyzing Citi Bike Data
# ## by Shakhawat Hassan

# ## Shakhawat Hassan

# ## Introduction
# Citi bike is a public bike-sharing system and it is now owned by Lyft. Lyft operates Citi bike around New York City, and Jersey City, NJ. Citi bike bicycle sharing system was opened in May 2013 with 332 stations and 6,000 bikes. Due to the high volume of growth rate, Citi bike is expanding around the five boroughs of NYC. As of July 2017, there are 130,000 annual subscribers. The average is 48,315 rides per day in 2018. Rides per day/week/month/year are still growing rapidly.
# * Due to staying home order for Covid-19 there might be a decrease in rides per day.

# In[1]:


from timeit import default_timer as timer
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# suppress warnings from final output
import warnings
warnings.simplefilter("ignore")


# ## Preliminary Wrangling

# In[2]:


df = pd.read_csv('2020feb_citibike.csv')


# In[3]:


df.shape 


# In[4]:


#22,962 rows and 15 columns


# In[5]:


df.isnull().sum()       #nulls checking


# In[6]:


df.duplicated().sum()          #There is no duplicate


# In[7]:


df.tail()


# In[8]:


df['bikeid'].count()


# #### 22962 users who rented citi bikes in February 2020

# In[9]:


df['tripduration'].nlargest() 


# In[10]:


df.info() #data type checking


# In[11]:


df.mean() #mean checking


# In[12]:


df.describe() #stats checking


# In[13]:


# 'start station id' , 'end station id', bikeid, should to be string, "birth year' is timedate, 'gender' is categorical


# In[14]:


top_start_station = df['start station name'].value_counts().head(15)
top_start_station


# Top 3 start stations are Grove St PATH, Sip Ave, and Hamilton Park.

# In[15]:


top_end_station = df['end station name'].value_counts().head(15)
top_end_station


# Top 3 end stations are Grove St PATH, Sip Ave, and Harborside.

# In[16]:


df.std() #standard deviation


# In[17]:


df.head()


# ### Exploratory Analysis

# There are 22,962 users who rented Citi bikes in February 2020. The highest trip was 1,495,458 secs and it was taken by the bike id of 42228. The average trip duration is 665 seconds. Top 3 start stations are Grove St PATH, Sip Ave, and Hamilton Park. Top 3 end stations are Grove St PATH, Sip Ave, and Harborside. Some data types should be cleaned. 
# 

# ## Data Wrangling

# In[18]:


df.info()


# In[19]:


df.head()


# In[20]:


#drop unnecessary columns
df.drop(['start station id', 'start station latitude',
         'start station longitude', 'end station id', 'end station latitude', 'end station longitude'], axis =1, inplace=True)


# In[21]:


#Change columns' names
df.rename(columns={'tripduration': 'duration_sec' , 'starttime': 'start_time', 
                   'start station name':'start_station_name', 'end station name':'end_station_name',
                   'stoptime': 'stop_time', 'bikeid':'bike_id', 'usertype':'user_type', 'birth year': 'birth_year'}, inplace =True)


# In[22]:


df.columns


# #### Fixing data types

# In[23]:


# string to datetime
df['start_time'] = pd.to_datetime(df['start_time'].str.strip())
df['stop_time'] = pd.to_datetime(df['stop_time'].str.strip())


# In[24]:


# integer to object
df['bike_id'] = df['bike_id'].apply(str)


# In[25]:


# integer to category
df['gender'] = df['gender'].astype('category')


# #### Extract weekdays, day, and hour from the dataset

# In[26]:


#Extract start month name
df['start_time_month_name'] = df ['start_time'].dt.strftime('%B')


# In[27]:


#extract start time weekdays
df['start_time_weekday'] = df ['start_time'].dt.strftime('%a')


# In[28]:


#extract start time day
df['start_time_day'] = df['start_time'].dt.day.astype(int)


# In[29]:


#extract start time hour
df['start_time_hour'] = df['start_time'].dt.hour


# df.drop(['start_time'], axis =1, inplace=True)

# #### Calculate the rider's age 

# In[30]:


#calucaling the rider's age\
df['rider_age'] = 2020 - df['birth_year']


# In[31]:


df.info()


# In[32]:


df.head(10)


# ### Data Wrangling Summary
# - drop unnecessary columns
# #### Fixing data types
# - string to datetime
# - integer to object
# - integer to category
# 

# ### What is the structure of your dataset?
# The dataset has 22,962 rows and columns. This means there were 22,962 rides that have been taken in the month of February 2020.
# 
# 
# - trip_duration: total time in a single trip (seconds)
# - start_time/stop_time: start or end time for a trip
# - start_station_name/end_station_name: name of the location where the trip was taken or ended.
# - bike_id : id number for a bike
# - user_type: (Customer = 24-hour pass or 3-day pass user; Subscriber = Annual Member)
# - birth_year : birth year for a customer
# - Gender (Zero=unknown; 1=male; 2=female)

# ### What is/are the main feature(s) of interest in your dataset?
# I am interested in the following
# - highest trip duration
# - what kind of a user took the most ride
# - which gender took the most rides
# - top start/end location's names

# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# - trip_duration: total time in a single trip (seconds)
# - user_type: (Customer = 24-hour pass or 3-day pass user; Subscriber = Annual Member)
# - birth_year : birth year for a customer
# - Gender (Zero=unknown; 1=male; 2=female)

# ## Univariate Exploration

# I'm going to analyze the riders' total trips by their gender, age, and user type. Next, I look into which stations are most popular.

# In[33]:


df.head()


# ### Trip Duration

# In[34]:


df['duration_sec'].describe()


# #### The max trip duration is 1,495, 458 seconds which is unusal

# #### Trip duration (seconds) is not normally distributed!
# - IQR : Q3 - Q1 = 517 - 226 = 291 seconds
# - Upper Whisker Bound: (1.5*IQR)+ Q3 = 436.5 + 517 = 953.5 seconds

# #### Statistic analysis
# - About 75% of the time, riders spend on bicycles about 517 seconds or 517/60 = 8.67 minutes
# - Riders take bikes at least for 61 seconds
# - On average, 665.69 seconds which means 11 minutes and 9 seconds
# 

# In[35]:


df[df['duration_sec'] >=300000]


# #### The max trip duration is 1,495, 458 seconds which is unusal. Let's remove this outlier

# In[36]:


#only keeping trip duration below 500,000 seconds
df = df[df['duration_sec'] <=300000]


# In[37]:


df[df['duration_sec'] >=300000]


# In[38]:


# outlier has been removed


# In[39]:


# Save the clean data
df.to_csv('clean_master_file.csv', index = False)


# #### Performing Normal Distribution

# In[40]:


log_binsize = 0.025
bins = 10 ** np.arange(0, np.log10(df['duration_sec'].max())+log_binsize, log_binsize)


plt.figure(figsize = [15,10])
plt.hist(data = df, x = 'duration_sec', bins =bins)
plt.xscale('log')
plt.xticks([50, 100, 500, 1000, 10000, 5000, 10000, 100000, 300000], [50, 100, 500, 1000, 10000, 5000, 10000, 100000, 300000])
plt.xlabel('Duration (secs)')
plt.ylabel('total trips')
plt.title('Normal Distribution of Trip Duration')


# Trip duration is now normally distributed.

# #### Rider's age

# In[41]:


df.rider_age.describe()


# In[42]:


#### a rider's age 132! this is unsual. Let's remove this outliers.


# In[43]:


df[df['rider_age'] >=75]


# In[44]:


#drop ages more than 75
df = df[df['rider_age'] <=75]


# In[45]:


df.rider_age.describe()


# ### Rider Age

# In[46]:


df['rider_age'].describe()


# In[47]:


plt.hist(data = df, x = 'rider_age')
plt.xlabel('rider_age')
plt.ylabel('total trips')
plt.title('Total trips vs Age')


# #### Most riders born between 1975 and 1990 and rheir ages between 30 and 40

# ### User Type

# #### Q. Does the above depend on if a user is a subscriber or customer?

# In[48]:


base_color = sb.color_palette()[0]
user_order = df['user_type'].value_counts().index
sb.countplot(data= df, x = 'user_type', color = base_color, order = user_order)
plt.xlabel('user type')
plt.ylabel('Total trips')
plt.title('Total trips vs User Type')


# #### Subscribers are renting more bikes than regular customers

# ### Gender

# In[49]:


base_color = sb.color_palette()[0]
gender_order = df['gender'].value_counts().index
sb.countplot(data= df, x = 'gender', color = base_color, order = gender_order)
plt.xticks(rotation =90)
plt.xlabel('gender')
plt.ylabel('Total trips')
plt.title('Total trips vs gender')


# #### Most riders are males

# ### Top 10 Start Stations

# In[50]:


top_start_station.plot.bar(figsize = (15,15))
plt.xticks(rotation =90)
plt.xlabel('Start station names')
plt.ylabel('Total number of times bikes rented')
plt.title('Top Ten Start Station')


# #### Visualization of top 10 start stations.
# Grove St PATH, Sip ave, and Hamilton Park places are the most popular stations where people get their bikes.

# ### Top 10 End Stations

# In[51]:


top_end_station.plot.bar(figsize = (15,15))
plt.xticks(rotation =90)
plt.xlabel('End station names')
plt.ylabel('Total number of times bikes rented')
plt.title('Top Ten End Station')


# #### Visualization of top 10 end stations.
# Grove St PATH, Sip ave, and Harborside places are the most popular stations where people return their bikes.

# #### Q. How long does the average trip take?
# 

# In[52]:


df['duration_sec'].mean()


# #### The average trip takes 660 seconds which equals to 10 minutes

# In[53]:


top_trips = df.nlargest(10, ['duration_sec'])
top_trips


# ## Bivariate Exploration
# 

# Here, I will visualize two variables to see the differences or similarities between them. Therefore, I will be able to see the trends.
#visualization of two variables
Quantitative vs Quantitative = Scatterplots
Quantitative vs Qualitative  = Violin plots
# ### Trip Duration vs Age

# In[54]:


# let's do analysis on two variables (trip duration vs age)
plt.figure(figsize = [15,10])
sb.regplot(data = df, x = 'rider_age', y = 'duration_sec')
plt.axis([10, 75, 500, 20000])
plt.title('Trip duration vs Age')
plt.xlabel('Rider Age')
plt.ylabel('Duration (sec)')


# #### Those riders who are between 35 and 50 years old, take bikes for a long period of time.

# In[55]:


df.info()


# ### Rider Age vs Gender

# In[56]:


plt.figure(figsize = [15, 10])
base_color = sb.color_palette()[0]

# left plot: violin plot
plt.subplot(1, 2, 1)
ax1 = sb.violinplot(data = df, x = 'gender', y = 'rider_age', color = base_color)

# right plot: box plot
plt.subplot(1, 2, 2)
sb.boxplot(data = df, x = 'gender', y = 'rider_age', color = base_color)
plt.ylim(ax1.get_ylim()) 


# #### The median age for both Males (1) and females (2) is 35 while the unknown gender is 50.

# #### Plotting Summary Statistic

# In[57]:


plt.figure(figsize = [10, 5])
base_color = sb.color_palette()[0]
sb.violinplot(data = df, x = 'gender', y = 'rider_age', color = base_color,
              inner = 'quartile')


# #### The median age for both Males (1) and females (2) is 35 while the unknown gender is 50.

# ### Rides Taken During the Week  by Gender

# In[58]:


plt.figure(figsize = [10, 5])
ax = sb.countplot(data = df, x = 'start_time_weekday', hue = 'gender')
ax.legend(loc = 2, ncol = 3, framealpha = 1, title = 'Gender')
plt.title('Rides during the Week by Gender')
plt.xlabel('Days')
plt.ylabel('Total Rides')


# #### Most males and females take Citi bikes on Wednesday or Tuesday while unknown gender takes bikes on Saturday or Sunday.

# #### I have observed very interesting relationships between rider's age and gender, and total rides taken during the week by gender.
# - Most males and females tend to take Citi bikes on Wednesday. 
# - Those riders who are between 35 and 50 years old, take bikes for a long period of time.

# ## Multivariate Exploration

# Visualizations of three of more variables
# - three numeric variables
# - two numeric variables and one categorical variable
# - one numeric variable and two categorical variables
# - three categorical variables

# ### Riders' Age vs User Type by Gender

# In[59]:


plt.figure(figsize = [10, 10])
g = sb.FacetGrid(data = df, col = 'gender', size = 4)
g.map(sb.boxplot, 'user_type', 'rider_age')


# Most males and females are subscribers who are between 30 and 45, and the median age is 35. On the other hand, the unknown gender is the customer (age 40-50). 

# ### Riders' Age and Gender by User Type

# In[60]:


plt.figure(figsize = [10, 10])
g = sb.FacetGrid(data = df, col = 'user_type', size = 4)
g.map(sb.boxplot, 'gender', 'rider_age')


# Both females and males prefer to get the annual membership.

# ### Weekly usage of Citi Bike per user type and gender

# In[61]:


g = sb.catplot(data= df, x = 'start_time_weekday', col = 'user_type', hue= 'gender', kind = 'count', sharey= False)


# Customers tend to take bike on Saturday and Sunday while subscribers tend to take bike on Wednesday

# ### Hourly usage of Citi Bike per user type and gender

# In[62]:


g = sb.catplot(data= df, x = 'start_time_hour', col = 'user_type', hue= 'gender', kind = 'count', sharey= False)


# Male customers take more bikes from 12 to 8 pm while male subscribers take bikes from 7 pm to 9 pm.

# - https://www.citibikenyc.com/system-data
# - https://s3.amazonaws.com/tripdata/index.html
# - https://en.wikipedia.org/wiki/Citi_Bike

# In[ ]:




