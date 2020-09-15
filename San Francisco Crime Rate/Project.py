# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:35:05 2019

@author: KTC
"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

f = pd.read_csv('Police_Department_Incident_Reports__Historical_2003_to_May_2018.csv')    #Preparing for original data
f.drop(['IncidntNum'], axis = 1, inplace = True)
f.drop(['Descript'], axis = 1, inplace = True)                                            #Drop all unneccessary columns
f.drop(['DayOfWeek'], axis = 1, inplace = True)
f.drop(['Address'], axis = 1, inplace = True)
f.drop(['X'], axis = 1, inplace = True)
f.drop(['Y'], axis = 1, inplace = True)
f.drop(['Location'], axis = 1, inplace = True)
f.drop(['PdId'], axis = 1, inplace = True)

f.rename(columns={'Category':'Incident'}, inplace = True)
f['Year'] = f['Date'].str[-4:]                                                           #Get info about Year





q_1 = f.groupby(f['Year']).agg({'Incident':'size'})    #Group by Year and count the number of incident
q_1.drop(['2018'], inplace = True)                     #Data from 2018 is not a whole data, thus we drop data from 2018
#print(q1)
#print(type(q1))    #Type is Dataframe

q_1.plot(grid = True, title = 'The Number of All Incident in SF from 2003 to 2017')   #Show Figure
plt.savefig('Question1.pdf', dpi = 1000)





q_1_2 = f.groupby([f['Year'], f['Incident']])
q_1_2 = q_1_2.agg(({'Incident':'size'}))
q_1_2.drop(['2018'], inplace = True)                                   #Data in 2018 is incompleted(only Jan-Apr available)
q_1_2.rename(columns={'Incident':'Number'}, inplace = True)
q_1_2 = q_1_2.reset_index()
q_1_2 = q_1_2.set_index('Incident')
q_1_2.drop(['BRIBERY'], inplace = True)    #Too many types of incident, we only analyze most common incidents which has relatively high numeber
q_1_2.drop(['ARSON'], inplace = True)
q_1_2.drop(['BAD CHECKS'], inplace = True)
q_1_2.drop(['DRIVING UNDER THE INFLUENCE'], inplace = True)
q_1_2.drop(['EMBEZZLEMENT'], inplace = True)
q_1_2.drop(['EXTORTION'], inplace = True)
q_1_2.drop(['FAMILY OFFENSES'], inplace = True)
q_1_2.drop(['GAMBLING'], inplace = True)
q_1_2.drop(['KIDNAPPING'], inplace = True)
q_1_2.drop(['LOITERING'], inplace = True)
q_1_2.drop(['PORNOGRAPHY/OBSCENE MAT'], inplace = True)
q_1_2.drop(['RUNAWAY'], inplace = True)
q_1_2.drop(['SEX OFFENSES, NON FORCIBLE'], inplace = True)
q_1_2.drop(['SUICIDE'], inplace = True)
q_1_2.drop(['TREA'], inplace = True)
q_1_2.drop(['LIQUOR LAWS'], inplace = True)
q_1_2.drop(['DISORDERLY CONDUCT'], inplace = True)
q_1_2.drop(['DRUNKENNESS'], inplace = True)
q_1_2.drop(['SECONDARY CODES'], inplace = True)
q_1_2.drop(['SEX OFFENSES, FORCIBLE'], inplace = True)
q_1_2.drop(['STOLEN PROPERTY'], inplace = True)
q_1_2.drop(['TRESPASS'], inplace = True)
q_1_2.drop(['WEAPON LAWS'], inplace = True)
q_1_2.drop(['FORGERY/COUNTERFEITING'], inplace = True)
q_1_2.drop(['PROSTITUTION'], inplace = True)
q_1_2.drop(['RECOVERED VEHICLE'], inplace = True)
q_1_2.drop(['FRAUD'], inplace = True)
q_1_2.drop(['MISSING PERSON'], inplace = True)
q_1_2.drop(['WARRANTS'], inplace = True)
q_1_2.drop(['DRUG/NARCOTIC'], inplace = True)
q_1_2.drop(['OTHER OFFENSES'], inplace = True)  #It is unclear what type of incident in OTHER OFFENSES, so we decided to remove it
q_1_2.drop(['NON-CRIMINAL'], inplace = True)    #Since it's NON-CRIMINAL, we don't want to analzye it
q_1_2 = q_1_2.reset_index()
 
sns.catplot(x = 'Year', y = 'Number', hue = 'Incident', kind = 'bar', data = q_1_2, height = 7.5, aspect = 1)   #Show Figure
plt.savefig('Question1_2.pdf', dpi = 1200)





q_1_3 = q_1_2.set_index('Incident')             # LARCENY/THEFT is the most commonm type of incident, let's take a close at other type of incidents
q_1_3.drop(['LARCENY/THEFT'], inplace = True)   #Remove LARCENY/THEFT
q_1_3 = q_1_3.reset_index()

sns.catplot(x = 'Year', y = 'Number', hue = 'Incident', kind = 'bar', data = q_1_3, height = 7.5, aspect = 1)
plt.savefig('Question1_3.pdf', dpi = 2000)





f['Time'] = f['Time'].apply(lambda x: datetime.strptime(x, '%H:%M'))    #Transfer str to datetime
q_2 = pd.DataFrame(f['Time'], columns = ['Time'])                       #Create a new Dataframe for question2
q_2['Totoal Number of Incidents'] = 1                                   #If there is incident, mark as 1
q_2 = q_2.set_index('Time')
q_2 = q_2.resample('120min', closed = 'left', label = 'left').sum()     #Downsampling to 2 Hours Frequency
q_2 = q_2.reset_index()
q_2['Time'] = q_2['Time'].apply(lambda x : x.strftime('%Y-%m-%d %H:%M:%S'))  #Transfer to Str
q_2['Time'] = q_2['Time'].apply(lambda x : x[-8:])                           #Only grabing Hour and Minutes info
q_2 = q_2.set_index('Time')
#print(q_2)
#print(type(q_2))

q_2.plot(grid = True, title = 'Total Number of Incident in Different Periods')   #Show Figure
plt.savefig('Question2.pdf', dpi = 2000)





q_2_2 = pd.DataFrame(f, columns = ['Time', 'Incident'])  #Grab Info about Time and types of Incidents
q_2_2['Totoal Number of Incidents'] = 1                  #If there is incident, mark as 1
q_2_2 = q_2_2.set_index('Time')                          #Setting Time as Index, So we can change timestamp to period
q_2_2 = q_2_2.to_period('H')                             #Change to period
q_2_2 = q_2_2.reset_index()
q_2_2['Time'] = q_2_2['Time'].apply(lambda x : x.strftime('%Y-%m-%d %H:%M:%S'))
q_2_2['Time'] = q_2_2['Time'].apply(lambda x : x[-8:])
q_2_2 = q_2_2.groupby([q_2_2['Time'], q_2_2['Incident']]).agg({'Totoal Number of Incidents':'sum'})  #Check most common type of Incidents in different time period(1 hour frequency) 
q_2_2.sort_values(['Totoal Number of Incidents'],ascending=False,inplace=True)
q_2_2 = q_2_2.reset_index()
q_2_2 = q_2_2.groupby([q_2_2['Time'], f['Incident']], group_keys=False).apply(lambda x: x.sort_values('Totoal Number of Incidents', ascending=False))
#print(q_2_2)

q_2_2.to_csv ('Question2_2.csv' , encoding = 'utf-8', index = False)   #Since the most common types of incidents in each time period is different, it's not suitable to visualize the data, instead we ouput a CSV file to help us to analyze.




q_3 = f.groupby([f['PdDistrict'], f['Incident']])
q_3 = q_3.agg(({'Incident':'size'}))
q_3.rename(columns={'Incident':'Number'}, inplace = True)
q_3 = q_3.reset_index()
q_3 = q_3.set_index('Incident')
q_3.drop(['BRIBERY'], inplace = True)    #Too many types of incident, we only analyze most common incidents which has relatively high numeber
q_3.drop(['ARSON'], inplace = True)
q_3.drop(['BAD CHECKS'], inplace = True)
q_3.drop(['DRIVING UNDER THE INFLUENCE'], inplace = True)
q_3.drop(['EMBEZZLEMENT'], inplace = True)
q_3.drop(['EXTORTION'], inplace = True)
q_3.drop(['FAMILY OFFENSES'], inplace = True)
q_3.drop(['GAMBLING'], inplace = True)
q_3.drop(['KIDNAPPING'], inplace = True)
q_3.drop(['LOITERING'], inplace = True)
q_3.drop(['PORNOGRAPHY/OBSCENE MAT'], inplace = True)
q_3.drop(['RUNAWAY'], inplace = True)
q_3.drop(['SEX OFFENSES, NON FORCIBLE'], inplace = True)
q_3.drop(['SUICIDE'], inplace = True)
q_3.drop(['TREA'], inplace = True)
q_3.drop(['LIQUOR LAWS'], inplace = True)
q_3.drop(['DISORDERLY CONDUCT'], inplace = True)
q_3.drop(['DRUNKENNESS'], inplace = True)
q_3.drop(['SECONDARY CODES'], inplace = True)
q_3.drop(['SEX OFFENSES, FORCIBLE'], inplace = True)
q_3.drop(['STOLEN PROPERTY'], inplace = True)
q_3.drop(['TRESPASS'], inplace = True)
q_3.drop(['WEAPON LAWS'], inplace = True)
q_3.drop(['FORGERY/COUNTERFEITING'], inplace = True)
q_3.drop(['PROSTITUTION'], inplace = True)
q_3.drop(['RECOVERED VEHICLE'], inplace = True)
q_3.drop(['FRAUD'], inplace = True)
q_3.drop(['MISSING PERSON'], inplace = True)
q_3.drop(['WARRANTS'], inplace = True)
q_3.drop(['DRUG/NARCOTIC'], inplace = True)
q_3.drop(['OTHER OFFENSES'], inplace = True)  #It is unclear what type of incident in OTHER OFFENSES, so we decided to remove it
q_3.drop(['NON-CRIMINAL'], inplace = True)    #Since it's NON-CRIMINAL, we don't want to analzye it
q_3 = q_3.reset_index()
#print(q_3)

sns.catplot(x = 'PdDistrict', y = 'Number', hue = 'Incident', kind = 'bar', data = q_3, height = 7.5, aspect = 1.5)   #Show Figure
plt.savefig('Question3.pdf', dpi = 2000)





q_3_2 = q_3.set_index('Incident')    #LARCENY/THEFT is the most commonm type of incident, let's take a close at other type of incidents
q_3_2.drop(['LARCENY/THEFT'], inplace = True)  #Remove LARCENY/THEFT
q_3_2 = q_3_2.reset_index()

sns.catplot(x = 'PdDistrict', y = 'Number', hue = 'Incident', kind = 'bar', data = q_3_2, height = 7.5, aspect = 1.5)   #Show Figure
plt.savefig('Question3_2.pdf', dpi = 2000)




q_3_3 = pd.DataFrame(f, columns = ['PdDistrict', 'Resolution'])      #Grab info about District and Resolution
total_incidents = q_3_3.groupby('PdDistrict')['Resolution'].size()   #Calculate total Incidents by District
q_3_3.set_index(['PdDistrict'], inplace = True)
q_3_3['Total_Incidents'] = total_incidents
q_3_3 = q_3_3.reset_index()
q_3_3['NONE'] = q_3_3['Resolution'].apply(lambda x: 0 if 'NONE' in x else 1)            #If not solved, mark as 0
q_3_3['UNFOUNDED'] = q_3_3['Resolution'].apply(lambda x: 0 if 'UNFOUNDED' in x else 1)  #If unfounded, mark as 0 
q_3_3['Solved or Not'] = q_3_3['NONE'] + q_3_3['UNFOUNDED'] - 1                         #If this incident solved mark as 1, otherwise mark as 0
q_3_3.drop(['NONE'], axis = 1, inplace = True)
q_3_3.drop(['UNFOUNDED'], axis = 1, inplace = True)
q_3_3.drop(['Resolution'], axis = 1, inplace = True)
q_3_3 = q_3_3.groupby(q_3_3['PdDistrict']).agg({'Solved or Not':'sum', 'Total_Incidents':'mean'})
q_3_3 = q_3_3.reset_index()
q_3_3['Resolution Rate'] = q_3_3['Solved or Not'] / q_3_3['Total_Incidents']
#print(total_incidents)
#print(q_3_3)

sns.catplot(x = 'PdDistrict', y = 'Resolution Rate', kind = 'bar', data = q_3_3, height = 7, aspect = 1.4)   #Show figure
plt.savefig('Question3_3.pdf', dpi = 2000)