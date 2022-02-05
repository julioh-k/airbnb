# Project Airbnb 
---
Airbnb is an online marketplace that connects people who want to rent out their homes with people who are looking for accommodations in specific locales.

## Problem statement

My project aims to help people set the best possible price for their Airbnb listing. Using different types of regression models, the model can predict what the most standard price will be for a new listing based on various factors, such as location, rooms, number of accommodations, etc.

Using natural language processing (NLP) analysis, I collected common verbs and text from a large sample of existing Airbnb listings in New York. This data is useful when predicting the price of listings.

With the help of time series models like ARIMA and fbprophet, analyze if there are any kind of trends or seasons in relation to the price. At the same try to predict the average price over time.

## Data gathering 

This data was taken from http://insideairbnb.com/get-the-data.html. The data behind the Inside Airbnb site is sourced from publicly available information from the Airbnb site. For the purpose of this project, we will use the data from New York City listings from February 6 and reservations from 2021 to 2022.

## Process

This project have three folders assets, code and data. 
- In the assets folder you will find the images obtained from the project. 
- In the code folder there are 3 notebooks: Airbnb notebook, EDA notebook and model notebook. These notebooks contain the description of the process.
- In the data folder contains all the datasets used for this project.

the code folder as mentioned above has 3 notebooks which have different functions.
- Airbnb notebook, is focused on data cleaning and organization.
- EDA notebook, continue the analysis through graphs and tables of listings, reviews and reservations.
- Model notebook, here we can see the results of the regression and time series models.

## Libraries an requirements needed it

The process of the process and models predictions have been created with jupyter lab using various python libraries.

The libraries used in this project are:
- Pandas
- Numpy
- Seaborn
- Plotly
- NLTK
- Sklearn
- Statsmodel
- Fbprophet

## Data dictionary 

Below we have the description of the 3 data obtained from the Airbnb website.

Description of the data [data description](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896).

- Listing: this dataframe contains all the information of the listings and the hosts, from the start of Airbnb in 2008 until the month of February 2022. This dataframe started with 38185 rows and 74 columns, and after cleaning and collecting the best features, it was reduced to 38149 rows with 32 columns.
- Review: This dataframe has the information of the reviews of the listings.
- Reservations_df: Here you will find the union of the two dataframes calendar_2021 and calendar_2022, with a size greater than 13000000 rows and 7 columns. this new dataframe contains the average price and the sum of the daily availability.

## Method


