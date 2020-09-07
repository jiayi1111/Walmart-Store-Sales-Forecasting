# Downcasting

I'll be downcasting the dataframes to reduce the amount of storage used by them and also to expidite the operations performed on them.

**Numerical Columns:**  
Depending on your environment, pandas automatically creates int32, int64, float32 or float64 columns for numeric ones. If you know the min or max value of a column, you can use a subtype which is less memory consuming. You can also use an unsigned subtype if there is no negative value.
Here are the different subtypes you can use:

 - int8 / uint8 : consumes 1 byte of memory, range between -128/127 or 0/255
 - bool : consumes 1 byte, true or false
 - float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
 - float32 / int32 / uint32 : consumes 4 bytes of memory, range between -2147483648 and 2147483647
 - float64 / int64 / uint64: consumes 8 bytes of memory
If one of your column has values between 1 and 10 for example, you will reduce the size of that column from 8 bytes per row to 1 byte, which is more than 85% memory saving on that column!

**Categorical Columns:**   
Pandas stores categorical columns as objects. 
One of the reason this storage is not optimal is that it creates a list of pointers to the memory address of each value of your column. 
For columns with low cardinality (the amount of unique values is lower than 50% of the count of these values), this can be optimized by forcing pandas to use a virtual mapping table where all unique values are mapped via an integer instead of a pointer. This is done using the category datatype.




# Data Manipulation

**Convert from wide to long format**  

In this case what the melt function is doing is that it is converting the sales dataframe which is in wide format to a long format. I have kept the id variables as id, item_id, dept_id, cat_id, store_id and state_id. They have in total 30490 unique values when compunded together. Now the total number of days for which we have the data is 1969 days. Therefore the melted dataframe will be having 30490x1969 i.e. 60034810 rows

Here's an example of conversion of a wide dataframe to a long dataframe.
<img src='https://pandas.pydata.org/pandas-docs/version/0.25.0/_images/reshaping_melt.png' style="width:600px;height:300px;">

```python
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()
 ```
**Combine the data**

Combine price data from prices dataframe and days data from calendar dataset.

 ```python
 df = pd.merge(df, calendar, on='d', how='left')
df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left') 
 
  ```


