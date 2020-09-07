# Walmart-Store-Sales-Forecasting Project

## Goal
The objective of the forecasting project is to advance the theory and practice of most accurate point forecastes and uncertainty distribution by identifying the methods that provide the unit sales of various products of different selling volumnes and prices for a horizon of 28 days that are organized in a hierarchical fashion, which contributes more valuable for the company.

## Data Summary
The unit sales of **3,049 products** in **3 product categories** (Hobbies, Foods and Household) and **7 product departments** across **10 stores located in three states( CA, TX and WI )**.

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

**Source:** https://www.kaggle.com/c/m5-forecasting-accuracy



# MileStone Report 
***

**A. Define the objective in business terms:** The objective is to come up with the right pricing algorithm that can we can use as a pricing recommendation to the users. 

**B. How will your solution be used?:** Allowing the users to see a suggest price before purchasing or selling will hopefully allow more transaction within Mercari's business. 

**C. How should you frame this problem?:** This problem can be solved using a supervised learning approach, and possible some unsupervised learning methods as well for clustering analysis. 

**D. How should performance be measured?:** Since its a regression problem, the evaluation metric that should be used is RMSE (Root Mean Squared Error). But in this case for the competition, we'll be using the 

**E. Are there any other data sets that you could use?:** To get a more accurate understanding and prediction for this problem, a potential dataset that we can gather would be more about the user. Features such as user location, user gender, and time could affect it.

## General Steps

1. Handle Missing Values — Replaced “missing” values with NA.

2. Lemmatization performed on item_description — Aiming to remove inflectional endings only and to return the base or dictionary form of a word

3. Label encoding has been performed on categorical values — Encode labels with value between 0 and n_classes-1.

4. Tokenization — Given a character sequence, tokenization is the task of chopping it up into pieces, called tokens and remove punctuation.

5. Maximum length of all sequences has been specified

6. Scaling performed on target variable (price)

7. Sentiment scored computed on item_description

8. Scaling performed on item description length as well



# Import Packages
***

```python
# general data manipulation
import pandas as pd
import numpy as np
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
```


```python
# general visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.gridspec import GridSpec
import plotly
from plotly import __version__
from plotly.offline import download_plotlyjs, plot, iplot
import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.figure_factory as ff
plotly.offline.init_notebook_mode()
import dash
import dash_core_components as dcc
import dash_html_components as html
```


```python
# forecast + modeling
from scipy import stats
from scipy.special import boxcox1p
import fbprophet
Prophet = fbprophet.Prophet
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import performance_metrics

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
```


```python
import warnings
warnings.filterwarnings("ignore")

from tqdm import tqdm
```
## Simple "Memory profilers" to see memory usage
```python
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)
```


```python
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
```
# Import Train / Test Data

```python

def read_data(PATH):
    print('Reading files...')
    calendar = pd.read_csv(f'{PATH}/calendar.csv',index_col='date',parse_dates=True)
    calendar = reduce_mem_usage(calendar)
    print('Calendar has {} rows and {} columns'.format(calendar.shape[0], calendar.shape[1]))
    sell_prices = pd.read_csv(f'{PATH}/sell_prices.csv')
    sell_prices = reduce_mem_usage(sell_prices)
    print('Sell prices has {} rows and {} columns'.format(sell_prices.shape[0], sell_prices.shape[1]))
    sales_train_validation = pd.read_csv(f'{PATH}/sales_train_validation.csv')
    print('Sales train validation has {} rows and {} columns'.format(sales_train_validation.shape[0], sales_train_validation.shape[1]))
    submission = pd.read_csv(f'{PATH}/sample_submission.csv')
    sales_train_evaluation = pd.read_csv(f'{PATH}/sales_train_evaluation.csv')
    print('sales_train_evaluation has {} rows and {} columns'.format(sales_train_evaluation.shape[0], sales_train_evaluation.shape[1]))
   
    return calendar, sell_prices, sales_train_validation, submission, sales_train_evaluation

calendar, sell_prices, sales_train_validation, submission, sales_train_evaluation = read_data("/Users/xujiayi/Desktop/walmart_forecasting")
```

# Data Preprocessing


```python
calendar["d"]=calendar["d"].apply(lambda x: int(x.split("_")[1]))
sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_validation"

```
## 1.Calculate weight for the level 12 series¶
```python
for day in tqdm(range(1858, 1886)):
    wk_id = list(calendar[calendar["d"]==day]["wm_yr_wk"])[0]
    wk_price_df = sell_prices[sell_prices["wm_yr_wk"]==wk_id]
    sales_train_validation = sales_train_validation.merge(wk_price_df[["sell_price", "id"]], on=["id"], how='inner')
    sales_train_validation["unit_sales_" + str(day)] = sales_train_validation["sell_price"] * sales_train_validation["d_" + str(day)]
    sales_train_validation.drop(columns=["sell_price"], inplace=True)
```
```Python
# add up all the new columns created horizontally for each of the level 12 series
sales_train_validation["dollar_sales"] = sales_train_validation[[c for c in sales_train_validation.columns if c.find("unit_sales")==0]].sum(axis=1)

# drop ennecessay columns to save space
sales_train_validation.drop(columns=[c for c in sales_train_validation.columns if c.find("unit_sales")==0], inplace=True)

# get the weight
sales_train_validation['weight'] = sales_train_validation['dollar_sales']/ sales_train_validation['dollar_sales'].sum()

sales_train_validation.drop(columns=["dollar_sales"], inplace=True)
df1 = sales_train_validation
df = sales_train_validation

df1["weight"] /= 12 # df1 topdowm
```

### 2 .Use the naive logic to make forecasts for each of the level 12 series (bu)
- All 0s
- Average through all history
- Same as previous 28 days
- Mean of previous 10, 20, 30, 40, 50, 60 days
- Average of same day for all previous weeks

#### Mean of history
```python
df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) <= 1885] +\
       ["id"]].set_index("id").transpose()

complete_historical_mean_df =\
    df[[c for c in df.columns if c.find("d_")==0 and int(c.split("_")[1]) <= 1885] +\
       ["id"]].set_index("id").transpose().mean().reset_index()
       
complete_historical_mean_df.head()

# Nothing is always 0
df[[c for c in df.columns if c.find("d_")==0]].sum(axis=1).min()      

def find_first_non_0(s):
    assert type(s) == np.ndarray
    return (s!=0).argmax(axis=0)
    
non_0_strt_arr = []
hist_arr = np.array(df[[c for c in df.columns if c.find("d_")==0]])
for i in tqdm(range(len(df))):
    non_0_strt_arr.append(find_first_non_0(hist_arr[i, :]))
    
df.head(1)
```

### 3. Infer round truth values, and weights for all the higher level series by aggregating (bu+td)

### 4. Calculalte RMSSE for all series using the equation

```python

h = 28
n = 1885
def rmsse(ground_truth, forecast, train_series, axis=1):
    # assuming input are numpy array or matrices
    assert axis == 0 or axis == 1
    assert type(ground_truth) == np.ndarray and type(forecast) == np.ndarray and type(train_series) == np.ndarray
    
    if axis == 1:
        # using axis == 1 we must guarantee these are matrices and not arrays
        assert ground_truth.shape[1] > 1 and forecast.shape[1] > 1 and train_series.shape[1] > 1
    
    numerator = ((ground_truth - forecast)**2).sum(axis=axis)
    if axis == 1:
        denominator = 1/(n-1) * ((train_series[:, 1:] - train_series[:, :-1]) ** 2).sum(axis=axis)
    else:
        denominator = 1/(n-1) * ((train_series[1:] - train_series[:-1]) ** 2).sum(axis=axis)
    return (1/h * numerator/denominator) ** 0.5
 ```

<img src = "http://i63.tinypic.com/14ccuv6.jpg" /img>

# Conclusion 
***

I am happy to have done this competition because it has opened up my mind into the realm of NLP and it showed me how much pre-processing steps are involved for text data. I learned the most common steps for text pre-processing and this allowed me to prepare myself for future work whenever I’m against text data again. Another concept that I really learned to value more is the choice of algorithms and how important computation is whenever you’re dealing with large datasets. It took me a couple of minutes to even perform some data visualizations and modeling. Text data is everywhere and it can get messy. Understanding the fundamentals on how to tackle these problems will definitely help me out in the future.


```python

```
