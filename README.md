# Walmart-Store-Sales-Forecasting Project

## Goal
The objective of the forecasting project is to advance the theory and practice of most accurate point forecastes and uncertainty distribution by identifying the methods that provide the unit sales of various products of different selling volumnes and prices for a horizon of 28 days that are organized in a hierarchical fashion, which contributes more valuable for the company.

## Data Summary


**The data**: We are working with **42,840 hierarchical time series.** The data were obtained in the 3 US states of California (CA), Texas (TX), and Wisconsin (WI). “Hierarchical” here means that data can be aggregated on different levels: item level, department level, product category level, and state level. The sales information reaches back from Jan 2011 to June 2016. In addition to the sales numbers, we are also given corresponding data on prices, promotions, and holidays. Note, that we have been warned that **most of the time series contain zero values.**

The data comprises **3049** individual products from ***3 categories and 7 departments, sold in 10 stores in 3 states.*** The hierachical aggregation captures the combinations of these factors. For instance, we can create 1 time series for all sales, 3 time series for all sales per state, and so on. The largest category is sales of all individual 3049 products per 10 stores for 30490 time series.

## The metrics:

The point forecast submission are being evaluated using the **Root Mean Squared Scaled Error (RMSSE)**, which is derived from the Mean Absolute Scaled Error (MASE) that was designed to be scale invariant and symmetric. In a similar way to the MASE, the RMSSE is scale invariant and symmetric, and measures the prediction error (i.e. forecast - truth) relative to a “naive forecast” that simply assumes that step i = step i-1. In contrast to the MASE, here both prediction error and naive error are scaled to account for the goal of estimating average values in the presence of many zeros.

The metric is computed for each time series and then averaged accross all time series including weights. The weights are proportional to the sales volume of the item, in dollars, to give more importance to high selling products. Note, that the weights are based on the last 28 days of the training data, and that those dates will be adjusted for the ultimate evaluation data, as confirmed by the organisers.

### Dataset Dictionary
#### File 1: calendar.csv
Contains the dates on which products are sold. The dates are in a yyyy/dd/mm format.

  - `date`: The date in a “y-m-d” format.  
  - `wm_yr_wk `: The id of the week the date belongs to.  
  - `weekday`: The type of the day (Saturday, Sunday, ..., Friday).  
  - `wday `: The id of the weekday, starting from Saturday.  
  - `month`: The month of the date.  
  - `year `: The year of the date.  
  - `event_name_1`: If the date includes an event, the name of this event.
  - `event_type_1`: If the date includes an event, the type of this event.
  - `event_name_2`: If the date includes a second event, the name of this event.
  - `event_type_2`: If the date includes a second event, the type of this event.
  - `snap_CA, snap_TX, and snap_WI`: A binary variable (0 or 1) indicating whether the stores of CA, TX or WI allow SNAP 3 purchases on the examined date. 1 indicates that SNAP purchases are allowed.

#### File 2: sales_train_validation.csv
Contains the historical daily unit sales data per product and store [d_1 - d_1913].

  - `item_id`: The id of the product.
  - `dept_id`: The id of the department the product belongs to.
  - `cat_id`: The id of the category the product belongs to.
  - `store_id`: The id of the store where the product is sold.
  - `state_id`: The State where the store is located.
  - `d_1, d_2, ..., d_i, ... d_1941`: The number of units sold at day i, starting from 2011-01-29.


#### File 3: sell_prices.csv
Contains information about the price of the products sold per store and date.

  - `store_id`: The id of the store where the product is sold.
  - `item_id`: The id of the product.
  - `wm_yr_wk`: The id of the week.
  - `sell_price`: The price of the product for the given week/store. The price is provided per week (average across seven days). If not available, this means that the product was not sold during the examined week. Note that although prices are constant at weekly basis, they may change through time (both training and test set).

#### File 4: submission.csv
Demonstrates the correct format for submission to the competition.

Each row contains an id that is a concatenation of an item_id and a store_id, which is either validation (corresponding to the Public leaderboard), or evaluation (corresponding to the Private leaderboard). You are predicting 28 forecast days (F1-F28) of items sold for each row. For the validation rows, this corresponds to d_1914 - d_1941, and for the evaluation rows, this corresponds to d_1942 - d_1969. (Note: a month before the competition close, the ground truth for the validation rows will be provided.)

#### File 5: sales_train_evaluation.csv
Available one month before the competition deadline. It will include sales for [d_1 - d_1941].


In this competition, we need to forecast the sales for [d_1942 - d_1969]. These rows form the evaluation set. The rows [d_1914 - d_1941] form the validation set, and the remaining rows form the training set. Now, since we understand the dataset and know what to predict, let us visualize the dataset.

**Work on supply and demand**




# MileStone Report 
***

**A. Define the objective in business terms:** The objective is to  to predict sales data provided by the retail giant Walmart 28 days into the future. 

**B. How will your solution be used?:** Improve decision-making about the future. Forecasts help sales with goal planning and help marketing with ad budgets and promotional strategies. Buyers use sales forecasts to plan their purchasing cycles. 

**C. How should you frame this problem?:** This problem can be solved using different approaches,such as statistical analysis, General forecasting models, Deep Learning Recurrent Neural Networks and Facebook Prophet library

**D. How should performance be measured?:** Since its a regression problem, the evaluation metric that should be used is RMSE (Root Mean Squared Error). But in this case for the requirement, we will use Root Mean Squared Scaled Error (RMSSE), which is derived from the Mean Absolute Scaled Error (MASE) that was designed to be scale invariant and symmetric. In a similar way to the MASE, the RMSSE is scale invariant and symmetric, and measures the prediction error (i.e. forecast - truth) relative to a “naive forecast” that simply assumes that step i = step i-1. In contrast to the MASE, here both prediction error and naive error are scaled to account for the goal of estimating average values in the presence of many zeros.

**E. Are there any other data sets that you could use?:** To get a more accurate understanding and prediction for this problem, a potential dataset that we can gather would be event and marketing plans. Features such as target products. Or External dataset of Natural Disasters


## General Steps

### Data manipulation

   1.Fetch the data
   2.Downcasting
   
### EDA 

1. Visual overview: interactive time series plots
   - individual item-level time series- random sample
   - All aggregate sales
   - Sales per State
   - Sales per Store & Category
   - Sales per Department
   - Seasonalities- global

2. Explanatory variables: Price and calendar
   - Calendar
   - Item Prices
   - Connection to time series data
   
3. Individual time series with explanantory varibles

4. Summary Statistics

### Feature Engineering

1. Label Encoding
2. Introduce Lags
3. Mean Encoding
4. Rolling Window Stats
5. Expanding Window Stats
6. Trends
7. Save the data

### Modeling and Perdiction


5. Modeling

   - Train/Val split
   - Naive approach
   - Moving average
   - Holt linear
   - Exponential smoothing
   - ARIMA
   - Prophet
   - LGBM

Loss for each model



