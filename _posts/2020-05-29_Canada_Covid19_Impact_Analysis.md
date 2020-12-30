---
title: "Canada Covid-19 Impact Analysis"
date: 2020-05-29
tags: [Data Analysis, Data Visulaization, Impact Analysis, Data Wrangling]
header:
  image: "/images/canada.jpg"
excerpt: "Data Analysis, Data Visualization, Impact Analysis, Data Wrangling"
mathjax: "true"
---



**Objective:** This notebook is a an attempt to analyze the impact of the Covid-19 on Canada and it's economy over the months by using the datasets produced by the Statistics Canada. Statistics Canada (French: Statistique Canada), formed in 1971, is the Canadian government agency commissioned with producing statistics to help better understand Canada, its population, resources, economy, society, and culture. 

We will be basically using three main indicators the GDP, CPI and the working hours and how they have been impacted by the covid-19 to analyze it's impact on the canadian economy. So before starting let's first try to understand what these indicators indicate:

GDP by industry: Gross Domestic Product (GDP) by industry is one of the three GDP series produced by the Canadian System of National Accounts (CSNA) of Statistics Canada. It is also known as the Output based GDP, because it sums the value added (output less intermediate consumption of goods and services) of all industries in Canada. A seasonally adjusted series is one from which seasonal movements have been eliminated. More information about this can be obtained from -https://unstats.un.org/unsd/nationalaccount/workshops/2009/ottawa/AC188-Bk4.PDF

CPI: The Consumer Price Index (CPI) is not a cost-of-living index. We could compute a cost-of-living index for an individual if we had complete information about that person's taste and spending habits. To do this for a large number of people, let alone the total population of Canada, is impossible. For this reason, regularly published price indexes are based on the fixed-basket concept rather than the cost-of-living concept. More information about this can be obtained from - https://www.statcan.gc.ca/eng/subjects-start/prices_and_price_indexes/consumer_price_indexes

Workhours by industry: Actual hours worked is the sum of hours actually worked by all employed persons in the reference week. A seasonally adjusted series is one from which seasonal movements have been eliminated. It should be noted that the seasonally adjusted series contain irregular as well as longer-term cyclical fluctuations. The seasonal adjustment program is a complicated computer program which differentiates between these seasonal, cyclical and irregular movements in a series over a number of years and, on the basis of past movements, estimates appropriate seasonal factors for current data. On an annual basis, the historic series of seasonally adjusted data are revised in light of the most recent information on changes in seasonality. More information about this can be obtained from - https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1410028901


# Reading the datasets

*Reading the 'GDP by industry.csv' dataset:*


```python
#reading and viewing the GDP by NAICS classified industries data set
gdp=pd.read_csv("../input/GDP by industry.csv")
gdp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>North American Industry Classification System (NAICS)</th>
      <th>Nov-19</th>
      <th>Dec-19</th>
      <th>Jan-20</th>
      <th>Feb-20</th>
      <th>Mar-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>All industries  [T001]4</td>
      <td>19,81,970</td>
      <td>19,87,713</td>
      <td>19,88,839</td>
      <td>19,90,433</td>
      <td>18,46,869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Goods-producing industries  [T002]4</td>
      <td>5,71,952</td>
      <td>5,72,822</td>
      <td>5,74,167</td>
      <td>5,75,858</td>
      <td>5,49,519</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Service-producing industries  [T003]4</td>
      <td>14,06,484</td>
      <td>14,11,249</td>
      <td>14,11,159</td>
      <td>14,11,203</td>
      <td>12,96,191</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Business sector industries  [T004]</td>
      <td>16,31,026</td>
      <td>16,36,516</td>
      <td>16,37,443</td>
      <td>16,39,850</td>
      <td>15,22,290</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Non-business sector industries  [T007]</td>
      <td>3,51,077</td>
      <td>3,51,371</td>
      <td>3,51,555</td>
      <td>3,50,824</td>
      <td>3,24,870</td>
    </tr>
  </tbody>
</table>
</div>



Gross domestic product (GDP) at basic prices, by industry, monthly (x 1,000,000).
Seasonally adjusted at annual rates.

As can be seen from the first five rows of the data set, we have a list of industries and their GDP for the months of Nov-2019, Dec-2019, Jan-2020, Feb-2020, Mar-2020.


```python
#This displays general information about the dataset with informations like the column names their data types 
#and the count of non-null values for every column.
gdp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 36 entries, 0 to 35
    Data columns (total 6 columns):
     #   Column                                                 Non-Null Count  Dtype 
    ---  ------                                                 --------------  ----- 
     0   North American Industry Classification System (NAICS)  36 non-null     object
     1   Nov-19                                                 36 non-null     object
     2   Dec-19                                                 36 non-null     object
     3   Jan-20                                                 36 non-null     object
     4   Feb-20                                                 36 non-null     object
     5   Mar-20                                                 36 non-null     object
    dtypes: object(6)
    memory usage: 1.8+ KB
    


```python
#displays the columns present in the dataset
gdp.columns
```




    Index(['North American Industry Classification System (NAICS)', 'Nov-19',
           'Dec-19', 'Jan-20', 'Feb-20', 'Mar-20'],
          dtype='object')



*Reading the 'CPI_monthly.csv' dataset:*


```python
#reading and viewing the CPI by product and product groups data set
cpi=pd.read_csv('../input/CPI_monthly.csv')
cpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Products and product groups3 4</th>
      <th>Dec-19</th>
      <th>Jan-20</th>
      <th>Feb-20</th>
      <th>Mar-20</th>
      <th>Apr-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>All-items</td>
      <td>136.4</td>
      <td>136.8</td>
      <td>137.4</td>
      <td>136.6</td>
      <td>135.7</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Food 5</td>
      <td>151.9</td>
      <td>153.5</td>
      <td>152.9</td>
      <td>152.8</td>
      <td>154.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Shelter 6</td>
      <td>146.3</td>
      <td>146.4</td>
      <td>146.7</td>
      <td>146.5</td>
      <td>146.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Household operations, furnishings and equipment</td>
      <td>122.9</td>
      <td>122.6</td>
      <td>123.3</td>
      <td>123.7</td>
      <td>124.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Clothing and footwear</td>
      <td>95.3</td>
      <td>95.8</td>
      <td>97.4</td>
      <td>99.0</td>
      <td>93.2</td>
    </tr>
  </tbody>
</table>
</div>



Consumer Price Index, monthly, not seasonally adjusted.

As can be seen from the first five rows of the dataset, we have a list of products and product groups and their monthly CPI for the months of  Dec-2019, Jan-2020, Feb-2020, Mar-2020, Apr-2020.


```python
#This displays general information about the dataset with informations like the column names their data types 
#and the count of non-null values for every column.
cpi.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 28 entries, 0 to 27
    Data columns (total 6 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   Products and product groups3 4  15 non-null     object 
     1   Dec-19                          15 non-null     float64
     2   Jan-20                          15 non-null     float64
     3   Feb-20                          15 non-null     float64
     4   Mar-20                          15 non-null     float64
     5   Apr-20                          15 non-null     float64
    dtypes: float64(5), object(1)
    memory usage: 1.4+ KB
    

*Reading the 'Workhours_by_industry.csv' dataset:*


```python
#reading and viewing the workhours by NAICS classified industries dataset
workhours=pd.read_csv('../input/Workhours_by_industry.csv')
workhours.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>North American Industry Classification System (NAICS)5</th>
      <th>Jan-20</th>
      <th>Feb-20</th>
      <th>Mar-20</th>
      <th>Apr-20</th>
      <th>May-20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Total actual hours worked, all industries 6</td>
      <td>6,24,926.40</td>
      <td>6,32,365.70</td>
      <td>5,36,710.10</td>
      <td>4,56,983.80</td>
      <td>4,85,951.20</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Goods-producing sector 7</td>
      <td>1,47,523.70</td>
      <td>1,50,569.10</td>
      <td>1,38,063.20</td>
      <td>1,04,590.10</td>
      <td>1,16,968.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Agriculture 8</td>
      <td>12,169.70</td>
      <td>12,208.40</td>
      <td>11,622.50</td>
      <td>10,908.50</td>
      <td>10,995.90</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Forestry, fishing, mining, quarrying, oil and ...</td>
      <td>13,081.30</td>
      <td>12,543.70</td>
      <td>12,476.70</td>
      <td>10,387.50</td>
      <td>11,342.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Utilities</td>
      <td>5,139.30</td>
      <td>5,217.20</td>
      <td>4,928.80</td>
      <td>4,764.70</td>
      <td>4,922.40</td>
    </tr>
  </tbody>
</table>
</div>



Actual hours worked at main job by industry, monthly, seasonally adjusted, last 5 months (x 1,000) 

As can be seen from the first five rows of the dataset, we have a list of industries and their monthly workhours for the months of Jan-2020, Feb-2020, Mar-2020, Apr-2020, May-2020.


```python
#This displays general information about the dataset with informations like the column names their data types 
#and the count of non-null values for every column.
workhours.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19 entries, 0 to 18
    Data columns (total 6 columns):
     #   Column                                                  Non-Null Count  Dtype 
    ---  ------                                                  --------------  ----- 
     0   North American Industry Classification System (NAICS)5  19 non-null     object
     1   Jan-20                                                  19 non-null     object
     2   Feb-20                                                  19 non-null     object
     3   Mar-20                                                  19 non-null     object
     4   Apr-20                                                  19 non-null     object
     5   May-20                                                  19 non-null     object
    dtypes: object(6)
    memory usage: 1.0+ KB
    

# Data Wrangling and Data Preparation

As can be seen from the above section that the columns contain the months and the rows have the industries (for the GDP and workhours dataset) or products (for the CPI dataset). But for easier analysis of the data we would like to have the industries/products across the columns of the dataframe and the months across the rows, thus in order to obtain that we will take transpose of the dataframes.


```python
#taking transpose of the gdp dataframe for easier analysis
gdp_t=gdp.transpose()
gdp_t.columns=gdp_t.iloc[0]
gdp_t.drop('North American Industry Classification System (NAICS)',inplace=True,axis=0)
gdp_t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>North American Industry Classification System (NAICS)</th>
      <th>All industries  [T001]4</th>
      <th>Goods-producing industries  [T002]4</th>
      <th>Service-producing industries  [T003]4</th>
      <th>Business sector industries  [T004]</th>
      <th>Non-business sector industries  [T007]</th>
      <th>Industrial production  [T010]4</th>
      <th>Non-durable manufacturing industries  [T011]4</th>
      <th>Durable manufacturing industries  [T012]4</th>
      <th>Information and communication technology sector  [T013]4</th>
      <th>Energy sector  [T016]4</th>
      <th>...</th>
      <th>Real estate and rental and leasing  [53]</th>
      <th>Professional, scientific and technical services  [54]</th>
      <th>Management of companies and enterprises  [55]</th>
      <th>Administrative and support, waste management and remediation services  [56]</th>
      <th>Educational services  [61]</th>
      <th>Health care and social assistance  [62]</th>
      <th>Arts, entertainment and recreation  [71]</th>
      <th>Accommodation and food services  [72]</th>
      <th>Other services (except public administration)  [81]</th>
      <th>Public administration  [91]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Nov-19</th>
      <td>19,81,970</td>
      <td>5,71,952</td>
      <td>14,06,484</td>
      <td>16,31,026</td>
      <td>3,51,077</td>
      <td>3,95,811</td>
      <td>91,319</td>
      <td>1,07,650</td>
      <td>96,205</td>
      <td>1,77,291</td>
      <td>...</td>
      <td>2,53,957</td>
      <td>1,20,424</td>
      <td>9,341</td>
      <td>52,501</td>
      <td>1,05,115</td>
      <td>1,41,459</td>
      <td>15,546</td>
      <td>45,016</td>
      <td>37,906</td>
      <td>1,33,964</td>
    </tr>
    <tr>
      <th>Dec-19</th>
      <td>19,87,713</td>
      <td>5,72,822</td>
      <td>14,11,249</td>
      <td>16,36,516</td>
      <td>3,51,371</td>
      <td>3,96,820</td>
      <td>91,971</td>
      <td>1,07,022</td>
      <td>96,268</td>
      <td>1,79,041</td>
      <td>...</td>
      <td>2,54,453</td>
      <td>1,20,454</td>
      <td>9,330</td>
      <td>52,484</td>
      <td>1,04,742</td>
      <td>1,41,916</td>
      <td>15,648</td>
      <td>44,934</td>
      <td>37,816</td>
      <td>1,34,235</td>
    </tr>
    <tr>
      <th>Jan-20</th>
      <td>19,88,839</td>
      <td>5,74,167</td>
      <td>14,11,159</td>
      <td>16,37,443</td>
      <td>3,51,555</td>
      <td>3,97,304</td>
      <td>92,684</td>
      <td>1,06,548</td>
      <td>96,493</td>
      <td>1,78,045</td>
      <td>...</td>
      <td>2,54,751</td>
      <td>1,20,895</td>
      <td>9,382</td>
      <td>52,648</td>
      <td>1,04,108</td>
      <td>1,42,264</td>
      <td>15,504</td>
      <td>44,595</td>
      <td>37,913</td>
      <td>1,34,701</td>
    </tr>
    <tr>
      <th>Feb-20</th>
      <td>19,90,433</td>
      <td>5,75,858</td>
      <td>14,11,203</td>
      <td>16,39,850</td>
      <td>3,50,824</td>
      <td>3,97,747</td>
      <td>92,242</td>
      <td>1,06,584</td>
      <td>96,390</td>
      <td>1,79,094</td>
      <td>...</td>
      <td>2,56,103</td>
      <td>1,21,195</td>
      <td>9,445</td>
      <td>52,525</td>
      <td>1,02,795</td>
      <td>1,42,637</td>
      <td>15,551</td>
      <td>44,429</td>
      <td>37,766</td>
      <td>1,35,089</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>18,46,869</td>
      <td>5,49,519</td>
      <td>12,96,191</td>
      <td>15,22,290</td>
      <td>3,24,870</td>
      <td>3,76,978</td>
      <td>91,243</td>
      <td>94,789</td>
      <td>94,933</td>
      <td>1,73,449</td>
      <td>...</td>
      <td>2,53,443</td>
      <td>1,10,803</td>
      <td>8,999</td>
      <td>45,224</td>
      <td>88,873</td>
      <td>1,26,866</td>
      <td>9,128</td>
      <td>28,054</td>
      <td>32,118</td>
      <td>1,28,957</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
#taking transpose of the cpi dataframe for easier analysis
cpi_t=cpi.transpose()
cpi_t.columns=cpi_t.iloc[0]
cpi_t.drop(['Products and product groups3 4'],axis=0,inplace=True)
cpi_t.dropna(inplace=True,axis=1)
cpi_t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Products and product groups3 4</th>
      <th>All-items</th>
      <th>Food 5</th>
      <th>Shelter 6</th>
      <th>Household operations, furnishings and equipment</th>
      <th>Clothing and footwear</th>
      <th>Transportation</th>
      <th>Gasoline</th>
      <th>Health and personal care</th>
      <th>Recreation, education and reading</th>
      <th>Alcoholic beverages, tobacco products and recreational cannabis</th>
      <th>All-items excluding food and energy 7</th>
      <th>All-items excluding energy 7</th>
      <th>Energy 7</th>
      <th>Goods 8</th>
      <th>Services 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dec-19</th>
      <td>136.4</td>
      <td>151.9</td>
      <td>146.3</td>
      <td>122.9</td>
      <td>95.3</td>
      <td>143.1</td>
      <td>165.8</td>
      <td>127.9</td>
      <td>113.2</td>
      <td>170</td>
      <td>130.8</td>
      <td>134.5</td>
      <td>158.3</td>
      <td>122.6</td>
      <td>150.1</td>
    </tr>
    <tr>
      <th>Jan-20</th>
      <td>136.8</td>
      <td>153.5</td>
      <td>146.4</td>
      <td>122.6</td>
      <td>95.8</td>
      <td>143.4</td>
      <td>166.4</td>
      <td>128.6</td>
      <td>113</td>
      <td>171.4</td>
      <td>131</td>
      <td>134.9</td>
      <td>158.6</td>
      <td>123.7</td>
      <td>149.7</td>
    </tr>
    <tr>
      <th>Feb-20</th>
      <td>137.4</td>
      <td>152.9</td>
      <td>146.7</td>
      <td>123.3</td>
      <td>97.4</td>
      <td>144</td>
      <td>163.1</td>
      <td>128.8</td>
      <td>116</td>
      <td>171.4</td>
      <td>132.1</td>
      <td>135.7</td>
      <td>156.6</td>
      <td>123.9</td>
      <td>150.8</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>136.6</td>
      <td>152.8</td>
      <td>146.5</td>
      <td>123.7</td>
      <td>99</td>
      <td>138.9</td>
      <td>134</td>
      <td>128.6</td>
      <td>116</td>
      <td>171.5</td>
      <td>132.2</td>
      <td>135.8</td>
      <td>140.6</td>
      <td>122</td>
      <td>151.1</td>
    </tr>
    <tr>
      <th>Apr-20</th>
      <td>135.7</td>
      <td>154</td>
      <td>146</td>
      <td>124.2</td>
      <td>93.2</td>
      <td>136.7</td>
      <td>113.6</td>
      <td>128.7</td>
      <td>115</td>
      <td>172.1</td>
      <td>131.8</td>
      <td>135.7</td>
      <td>128.3</td>
      <td>120.2</td>
      <td>151.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#taking transpose of the workhours dataframe for easier analysis
workhours_t=workhours.transpose()
workhours_t.columns=workhours_t.iloc[0]
workhours_t.drop('North American Industry Classification System (NAICS)5',inplace=True,axis=0)
workhours_t.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>North American Industry Classification System (NAICS)5</th>
      <th>Total actual hours worked, all industries 6</th>
      <th>Goods-producing sector 7</th>
      <th>Agriculture 8</th>
      <th>Forestry, fishing, mining, quarrying, oil and gas 9 10</th>
      <th>Utilities</th>
      <th>Construction</th>
      <th>Manufacturing</th>
      <th>Services-producing sector 11</th>
      <th>Wholesale and retail trade</th>
      <th>Transportation and warehousing</th>
      <th>Finance, insurance, real estate, rental and leasing</th>
      <th>Professional, scientific and technical services</th>
      <th>Business, building and other support services 12</th>
      <th>Educational services</th>
      <th>Health care and social assistance</th>
      <th>Information, culture and recreation</th>
      <th>Accommodation and food services</th>
      <th>Other services (except public administration)</th>
      <th>Public administration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan-20</th>
      <td>6,24,926.40</td>
      <td>1,47,523.70</td>
      <td>12,169.70</td>
      <td>13,081.30</td>
      <td>5,139.30</td>
      <td>53,612.80</td>
      <td>63,520.60</td>
      <td>4,77,402.80</td>
      <td>88,296.70</td>
      <td>37,584.40</td>
      <td>41,957.30</td>
      <td>54,789.70</td>
      <td>24,018.60</td>
      <td>39,620.20</td>
      <td>75,465.90</td>
      <td>23,030.10</td>
      <td>33,205.90</td>
      <td>25,864.80</td>
      <td>33,569.20</td>
    </tr>
    <tr>
      <th>Feb-20</th>
      <td>6,32,365.70</td>
      <td>1,50,569.10</td>
      <td>12,208.40</td>
      <td>12,543.70</td>
      <td>5,217.20</td>
      <td>55,112.80</td>
      <td>65,487.00</td>
      <td>4,81,796.50</td>
      <td>90,184.10</td>
      <td>37,784.40</td>
      <td>42,511.40</td>
      <td>53,964.10</td>
      <td>24,081.50</td>
      <td>39,392.20</td>
      <td>76,261.90</td>
      <td>23,991.20</td>
      <td>33,505.20</td>
      <td>26,430.60</td>
      <td>33,689.90</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>5,36,710.10</td>
      <td>1,38,063.20</td>
      <td>11,622.50</td>
      <td>12,476.70</td>
      <td>4,928.80</td>
      <td>48,179.30</td>
      <td>60,855.90</td>
      <td>3,98,646.90</td>
      <td>77,315.60</td>
      <td>33,109.80</td>
      <td>39,668.60</td>
      <td>49,648.40</td>
      <td>20,834.10</td>
      <td>28,027.70</td>
      <td>62,618.60</td>
      <td>16,639.70</td>
      <td>19,739.20</td>
      <td>20,541.70</td>
      <td>30,503.70</td>
    </tr>
    <tr>
      <th>Apr-20</th>
      <td>4,56,983.80</td>
      <td>1,04,590.10</td>
      <td>10,908.50</td>
      <td>10,387.50</td>
      <td>4,764.70</td>
      <td>32,185.40</td>
      <td>46,343.90</td>
      <td>3,52,393.80</td>
      <td>62,226.50</td>
      <td>27,337.90</td>
      <td>36,854.50</td>
      <td>47,230.90</td>
      <td>16,783.70</td>
      <td>30,110.90</td>
      <td>59,254.40</td>
      <td>15,001.60</td>
      <td>12,120.80</td>
      <td>13,672.20</td>
      <td>31,800.50</td>
    </tr>
    <tr>
      <th>May-20</th>
      <td>4,85,951.20</td>
      <td>1,16,968.00</td>
      <td>10,995.90</td>
      <td>11,342.00</td>
      <td>4,922.40</td>
      <td>38,314.70</td>
      <td>51,393.00</td>
      <td>3,68,983.20</td>
      <td>69,101.30</td>
      <td>27,222.40</td>
      <td>38,328.40</td>
      <td>47,651.80</td>
      <td>16,624.10</td>
      <td>32,935.70</td>
      <td>61,796.00</td>
      <td>15,087.10</td>
      <td>12,408.80</td>
      <td>15,473.90</td>
      <td>32,353.60</td>
    </tr>
  </tbody>
</table>
</div>



Next, we will check for the presence of any null values in any column of the dataframes.


```python
#checking for null values
gdp_t.isnull().sum()
```




    North American Industry Classification System (NAICS)
    All industries  [T001]4                                                        0
    Goods-producing industries  [T002]4                                            0
    Service-producing industries  [T003]4                                          0
    Business sector industries  [T004]                                             0
    Non-business sector industries  [T007]                                         0
    Industrial production  [T010]4                                                 0
    Non-durable manufacturing industries  [T011]4                                  0
    Durable manufacturing industries  [T012]4                                      0
    Information and communication technology sector  [T013]4                       0
    Energy sector  [T016]4                                                         0
    Industrial production (1950 definition)  [T017]4                               0
    Public Sector  [T018]4                                                         0
    Content and media sector  [T019]4                                              0
    All industries (except cannabis sector)  [T020]4                               0
    Cannabis sector  [T021]4                                                       0
    All industries (except unlicensed cannabis sector)  [T024]4                    0
    Agriculture, forestry, fishing and hunting  [11]                               0
    Mining, quarrying, and oil and gas extraction  [21]                            0
    Utilities  [22]                                                                0
    Construction  [23]                                                             0
    Manufacturing  [31-33]                                                         0
    Wholesale trade  [41]                                                          0
    Retail trade  [44-45]                                                          0
    Transportation and warehousing  [48-49]                                        0
    Information and cultural industries  [51]                                      0
    Finance and insurance  [52]                                                    0
    Real estate and rental and leasing  [53]                                       0
    Professional, scientific and technical services  [54]                          0
    Management of companies and enterprises  [55]                                  0
    Administrative and support, waste management and remediation services  [56]    0
    Educational services  [61]                                                     0
    Health care and social assistance  [62]                                        0
    Arts, entertainment and recreation  [71]                                       0
    Accommodation and food services  [72]                                          0
    Other services (except public administration)  [81]                            0
    Public administration  [91]                                                    0
    dtype: int64




```python
#checking for null values
cpi_t.isna().sum()
```




    Products and product groups3 4
    All-items                                                          0
    Food 5                                                             0
    Shelter 6                                                          0
    Household operations, furnishings and equipment                    0
    Clothing and footwear                                              0
    Transportation                                                     0
    Gasoline                                                           0
    Health and personal care                                           0
    Recreation, education and reading                                  0
    Alcoholic beverages, tobacco products and recreational cannabis    0
    All-items excluding food and energy 7                              0
    All-items excluding energy 7                                       0
    Energy 7                                                           0
    Goods 8                                                            0
    Services 9                                                         0
    dtype: int64




```python
#checking for null values
workhours_t.isnull().sum()
```




    North American Industry Classification System (NAICS)5
    Total actual hours worked, all industries 6               0
    Goods-producing sector 7                                  0
    Agriculture 8                                             0
    Forestry, fishing, mining, quarrying, oil and gas 9 10    0
    Utilities                                                 0
    Construction                                              0
    Manufacturing                                             0
    Services-producing sector 11                              0
    Wholesale and retail trade                                0
    Transportation and warehousing                            0
    Finance, insurance, real estate, rental and leasing       0
    Professional, scientific and technical services           0
    Business, building and other support services 12          0
    Educational services                                      0
    Health care and social assistance                         0
    Information, culture and recreation                       0
    Accommodation and food services                           0
    Other services (except public administration)             0
    Public administration                                     0
    dtype: int64



Thus there are no null values.

As we noticed that for the GDP and Workhours dataframes, the numerical columns are of object type but we would like to have them in float/int datatype for better analysis of the data, thus we will convert the datatype of these columns to 'float' from 'object'.

*Converting the datatype of the numerical columns of the GDP dataset*


```python
#changing the datatypes from object to float 
for x in gdp_t.iloc[:,:]:
    gdp_t[x]=gdp_t[x].apply(lambda y: float(y.replace(',','')))
    
gdp_t.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5 entries, Nov-19 to Mar-20
    Data columns (total 36 columns):
     #   Column                                                                       Non-Null Count  Dtype  
    ---  ------                                                                       --------------  -----  
     0   All industries  [T001]4                                                      5 non-null      float64
     1   Goods-producing industries  [T002]4                                          5 non-null      float64
     2   Service-producing industries  [T003]4                                        5 non-null      float64
     3   Business sector industries  [T004]                                           5 non-null      float64
     4   Non-business sector industries  [T007]                                       5 non-null      float64
     5   Industrial production  [T010]4                                               5 non-null      float64
     6   Non-durable manufacturing industries  [T011]4                                5 non-null      float64
     7   Durable manufacturing industries  [T012]4                                    5 non-null      float64
     8   Information and communication technology sector  [T013]4                     5 non-null      float64
     9   Energy sector  [T016]4                                                       5 non-null      float64
     10  Industrial production (1950 definition)  [T017]4                             5 non-null      float64
     11  Public Sector  [T018]4                                                       5 non-null      float64
     12  Content and media sector  [T019]4                                            5 non-null      float64
     13  All industries (except cannabis sector)  [T020]4                             5 non-null      float64
     14  Cannabis sector  [T021]4                                                     5 non-null      float64
     15  All industries (except unlicensed cannabis sector)  [T024]4                  5 non-null      float64
     16  Agriculture, forestry, fishing and hunting  [11]                             5 non-null      float64
     17  Mining, quarrying, and oil and gas extraction  [21]                          5 non-null      float64
     18  Utilities  [22]                                                              5 non-null      float64
     19  Construction  [23]                                                           5 non-null      float64
     20  Manufacturing  [31-33]                                                       5 non-null      float64
     21  Wholesale trade  [41]                                                        5 non-null      float64
     22  Retail trade  [44-45]                                                        5 non-null      float64
     23  Transportation and warehousing  [48-49]                                      5 non-null      float64
     24  Information and cultural industries  [51]                                    5 non-null      float64
     25  Finance and insurance  [52]                                                  5 non-null      float64
     26  Real estate and rental and leasing  [53]                                     5 non-null      float64
     27  Professional, scientific and technical services  [54]                        5 non-null      float64
     28  Management of companies and enterprises  [55]                                5 non-null      float64
     29  Administrative and support, waste management and remediation services  [56]  5 non-null      float64
     30  Educational services  [61]                                                   5 non-null      float64
     31  Health care and social assistance  [62]                                      5 non-null      float64
     32  Arts, entertainment and recreation  [71]                                     5 non-null      float64
     33  Accommodation and food services  [72]                                        5 non-null      float64
     34  Other services (except public administration)  [81]                          5 non-null      float64
     35  Public administration  [91]                                                  5 non-null      float64
    dtypes: float64(36)
    memory usage: 1.4+ KB
    

*Converting the datatype of the Workhours by industry dataset*


```python
#changing the datatypes from object to float 
for x in workhours_t.iloc[:,:]:
    workhours_t[x]=workhours_t[x].apply(lambda y: float(y.replace(',','')))
    
workhours_t.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 5 entries, Jan-20 to May-20
    Data columns (total 19 columns):
     #   Column                                                  Non-Null Count  Dtype  
    ---  ------                                                  --------------  -----  
     0   Total actual hours worked, all industries 6             5 non-null      float64
     1   Goods-producing sector 7                                5 non-null      float64
     2   Agriculture 8                                           5 non-null      float64
     3   Forestry, fishing, mining, quarrying, oil and gas 9 10  5 non-null      float64
     4   Utilities                                               5 non-null      float64
     5   Construction                                            5 non-null      float64
     6   Manufacturing                                           5 non-null      float64
     7   Services-producing sector 11                            5 non-null      float64
     8   Wholesale and retail trade                              5 non-null      float64
     9   Transportation and warehousing                          5 non-null      float64
     10  Finance, insurance, real estate, rental and leasing     5 non-null      float64
     11  Professional, scientific and technical services         5 non-null      float64
     12  Business, building and other support services 12        5 non-null      float64
     13  Educational services                                    5 non-null      float64
     14  Health care and social assistance                       5 non-null      float64
     15  Information, culture and recreation                     5 non-null      float64
     16  Accommodation and food services                         5 non-null      float64
     17  Other services (except public administration)           5 non-null      float64
     18  Public administration                                   5 non-null      float64
    dtypes: float64(19)
    memory usage: 800.0+ bytes
    

Now we have all the numerical columns in float type.

# Feature Engineering

For better analysis we would like to have a feature which will indicate whether there has been a decrease or increase in the GDP by industry or CPI by product/product group or Working hours by industry. Thus we will create a new feature 'Increased/Decreased Percentage' which will indicate the percentage increase/ decrease in the GDP/Working hours industry wise and CPI product wise.

*Feature Engineering for the GDP dataset:*


```python
#deriving a new feature which will represent the percentage of increase/decrease in GDP 
x=[]
for i in gdp_t.iloc[:,0:]:
    x.append(((gdp_t.loc['Mar-20',i]-gdp_t.loc['Nov-19',i])/gdp_t.loc['Nov-19',i])*100)
    
print(x)
```

    [-6.816500754300014, -3.9221822810305755, -7.841752910093539, -6.6667238903610375, -7.464744201414504, -4.758078982140466, -0.083224739648923, -11.947050627032048, -1.3221766020477106, -2.1670586775414433, -4.746877730379812, -9.40566221180477, -15.950209871182516, -6.807587021384287, -8.80374044931007, -6.821619494907251, -1.2629549288985298, -2.0452494968422514, -1.1317777026265359, -2.7808848461825266, -6.584866870077058, -3.6297818217395554, -8.830311083320307, -12.8820763248327, -2.787757304756765, -0.8106452287737964, -0.202396468693519, -7.989271241612967, -3.661278235734932, -13.860688367840613, -15.451648194834231, -10.316063311630932, -41.28393155795703, -37.67993602274747, -15.269350498601804, -3.737571287808665]
    

The negative values indicate that there has been a net decrease in the GDP from the month of Nov-2019 to Mar-2020, for all the industries listed in the dataset and this decrease in GDP varies across the industries.


```python
#appending the newly created feature to the gdp_t dataframe
x=pd.Series(x,name='Increased/Decreased Percentage',index=gdp_t.columns)
gdp_t=gdp_t.append(x,ignore_index=False)

gdp_t.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>North American Industry Classification System (NAICS)</th>
      <th>All industries  [T001]4</th>
      <th>Goods-producing industries  [T002]4</th>
      <th>Service-producing industries  [T003]4</th>
      <th>Business sector industries  [T004]</th>
      <th>Non-business sector industries  [T007]</th>
      <th>Industrial production  [T010]4</th>
      <th>Non-durable manufacturing industries  [T011]4</th>
      <th>Durable manufacturing industries  [T012]4</th>
      <th>Information and communication technology sector  [T013]4</th>
      <th>Energy sector  [T016]4</th>
      <th>...</th>
      <th>Real estate and rental and leasing  [53]</th>
      <th>Professional, scientific and technical services  [54]</th>
      <th>Management of companies and enterprises  [55]</th>
      <th>Administrative and support, waste management and remediation services  [56]</th>
      <th>Educational services  [61]</th>
      <th>Health care and social assistance  [62]</th>
      <th>Arts, entertainment and recreation  [71]</th>
      <th>Accommodation and food services  [72]</th>
      <th>Other services (except public administration)  [81]</th>
      <th>Public administration  [91]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Dec-19</th>
      <td>1.987713e+06</td>
      <td>572822.000000</td>
      <td>1.411249e+06</td>
      <td>1.636516e+06</td>
      <td>351371.000000</td>
      <td>396820.000000</td>
      <td>91971.000000</td>
      <td>107022.000000</td>
      <td>96268.000000</td>
      <td>179041.000000</td>
      <td>...</td>
      <td>254453.000000</td>
      <td>120454.000000</td>
      <td>9330.000000</td>
      <td>52484.000000</td>
      <td>104742.000000</td>
      <td>141916.000000</td>
      <td>15648.000000</td>
      <td>44934.000000</td>
      <td>37816.00000</td>
      <td>134235.000000</td>
    </tr>
    <tr>
      <th>Jan-20</th>
      <td>1.988839e+06</td>
      <td>574167.000000</td>
      <td>1.411159e+06</td>
      <td>1.637443e+06</td>
      <td>351555.000000</td>
      <td>397304.000000</td>
      <td>92684.000000</td>
      <td>106548.000000</td>
      <td>96493.000000</td>
      <td>178045.000000</td>
      <td>...</td>
      <td>254751.000000</td>
      <td>120895.000000</td>
      <td>9382.000000</td>
      <td>52648.000000</td>
      <td>104108.000000</td>
      <td>142264.000000</td>
      <td>15504.000000</td>
      <td>44595.000000</td>
      <td>37913.00000</td>
      <td>134701.000000</td>
    </tr>
    <tr>
      <th>Feb-20</th>
      <td>1.990433e+06</td>
      <td>575858.000000</td>
      <td>1.411203e+06</td>
      <td>1.639850e+06</td>
      <td>350824.000000</td>
      <td>397747.000000</td>
      <td>92242.000000</td>
      <td>106584.000000</td>
      <td>96390.000000</td>
      <td>179094.000000</td>
      <td>...</td>
      <td>256103.000000</td>
      <td>121195.000000</td>
      <td>9445.000000</td>
      <td>52525.000000</td>
      <td>102795.000000</td>
      <td>142637.000000</td>
      <td>15551.000000</td>
      <td>44429.000000</td>
      <td>37766.00000</td>
      <td>135089.000000</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>1.846869e+06</td>
      <td>549519.000000</td>
      <td>1.296191e+06</td>
      <td>1.522290e+06</td>
      <td>324870.000000</td>
      <td>376978.000000</td>
      <td>91243.000000</td>
      <td>94789.000000</td>
      <td>94933.000000</td>
      <td>173449.000000</td>
      <td>...</td>
      <td>253443.000000</td>
      <td>110803.000000</td>
      <td>8999.000000</td>
      <td>45224.000000</td>
      <td>88873.000000</td>
      <td>126866.000000</td>
      <td>9128.000000</td>
      <td>28054.000000</td>
      <td>32118.00000</td>
      <td>128957.000000</td>
    </tr>
    <tr>
      <th>Increased/Decreased Percentage</th>
      <td>-6.816501e+00</td>
      <td>-3.922182</td>
      <td>-7.841753e+00</td>
      <td>-6.666724e+00</td>
      <td>-7.464744</td>
      <td>-4.758079</td>
      <td>-0.083225</td>
      <td>-11.947051</td>
      <td>-1.322177</td>
      <td>-2.167059</td>
      <td>...</td>
      <td>-0.202396</td>
      <td>-7.989271</td>
      <td>-3.661278</td>
      <td>-13.860688</td>
      <td>-15.451648</td>
      <td>-10.316063</td>
      <td>-41.283932</td>
      <td>-37.679936</td>
      <td>-15.26935</td>
      <td>-3.737571</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>



So now we have a feature 'Increased/Decreased Percentage' in the dataframe which we can use for better analysis of the GDP trend over the time period.

*Feature Engineering for the CPI dataset:*


```python
#deriving a new feature which will represent the percentage of increase/decrease in CPI 
x=[]
for i in cpi_t.iloc[:,0:]:
    x.append(((cpi_t.loc['Apr-20',i]-cpi_t.loc['Dec-19',i])/cpi_t.loc['Dec-19',i])*100)
    
print(x)
```

    [-0.5131964809384288, 1.382488479262669, -0.20505809979494966, 1.0577705451586632, -2.2035676810073395, -4.472396925227118, -31.483715319662252, 0.6254886630179695, 1.5901060070671353, 1.2352941176470555, 0.7645259938837919, 0.8921933085501774, -18.951358180669615, -1.9575856443719346, 0.6662225183211192]
    

The negative values indicate that there has been a net decrease in the CPI for that particular product/product group from the month of Dec-2019 to Apr-2020 and the positive percentage values indicate a net increase in CPI for those products.


```python
#appending the newly created feature to the dataframe
x=pd.Series(x,name='Increased/Decreased Percentage',index=cpi_t.columns)
cpi_t=cpi_t.append(x,ignore_index=False)

cpi_t.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Products and product groups3 4</th>
      <th>All-items</th>
      <th>Food 5</th>
      <th>Shelter 6</th>
      <th>Household operations, furnishings and equipment</th>
      <th>Clothing and footwear</th>
      <th>Transportation</th>
      <th>Gasoline</th>
      <th>Health and personal care</th>
      <th>Recreation, education and reading</th>
      <th>Alcoholic beverages, tobacco products and recreational cannabis</th>
      <th>All-items excluding food and energy 7</th>
      <th>All-items excluding energy 7</th>
      <th>Energy 7</th>
      <th>Goods 8</th>
      <th>Services 9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Jan-20</th>
      <td>136.8</td>
      <td>153.5</td>
      <td>146.4</td>
      <td>122.6</td>
      <td>95.8</td>
      <td>143.4</td>
      <td>166.4</td>
      <td>128.6</td>
      <td>113</td>
      <td>171.4</td>
      <td>131</td>
      <td>134.9</td>
      <td>158.6</td>
      <td>123.7</td>
      <td>149.7</td>
    </tr>
    <tr>
      <th>Feb-20</th>
      <td>137.4</td>
      <td>152.9</td>
      <td>146.7</td>
      <td>123.3</td>
      <td>97.4</td>
      <td>144</td>
      <td>163.1</td>
      <td>128.8</td>
      <td>116</td>
      <td>171.4</td>
      <td>132.1</td>
      <td>135.7</td>
      <td>156.6</td>
      <td>123.9</td>
      <td>150.8</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>136.6</td>
      <td>152.8</td>
      <td>146.5</td>
      <td>123.7</td>
      <td>99</td>
      <td>138.9</td>
      <td>134</td>
      <td>128.6</td>
      <td>116</td>
      <td>171.5</td>
      <td>132.2</td>
      <td>135.8</td>
      <td>140.6</td>
      <td>122</td>
      <td>151.1</td>
    </tr>
    <tr>
      <th>Apr-20</th>
      <td>135.7</td>
      <td>154</td>
      <td>146</td>
      <td>124.2</td>
      <td>93.2</td>
      <td>136.7</td>
      <td>113.6</td>
      <td>128.7</td>
      <td>115</td>
      <td>172.1</td>
      <td>131.8</td>
      <td>135.7</td>
      <td>128.3</td>
      <td>120.2</td>
      <td>151.1</td>
    </tr>
    <tr>
      <th>Increased/Decreased Percentage</th>
      <td>-0.513196</td>
      <td>1.38249</td>
      <td>-0.205058</td>
      <td>1.05777</td>
      <td>-2.20357</td>
      <td>-4.4724</td>
      <td>-31.4837</td>
      <td>0.625489</td>
      <td>1.59011</td>
      <td>1.23529</td>
      <td>0.764526</td>
      <td>0.892193</td>
      <td>-18.9514</td>
      <td>-1.95759</td>
      <td>0.666223</td>
    </tr>
  </tbody>
</table>
</div>



*Feature Engineering for the Workhours dataset:*


```python
#deriving a new feature which will represent the percentage of increase/decrease in CPI 
x=[]
for i in workhours_t.iloc[:,0:]:
    x.append(((workhours_t.loc['May-20',i]-workhours_t.loc['Jan-20',i])/workhours_t.loc['Jan-20',i])*100)
    

print(x)  
```

    [-22.238650823520977, -20.71240078712777, -9.645266522593007, -13.296079135865696, -4.220419123226908, -28.53441715411246, -19.092388925797298, -22.710298305749355, -21.739657314486266, -27.569949234256764, -8.64903127703642, -13.027813621903379, -30.786557084925853, -16.871444364238446, -18.114009108749773, -34.48964615872271, -62.63073730873129, -40.173904302372335, -3.621176554698946]
    

The negative values indicate that there has been a net decrease in the working hours from the month of Jan-2020 to May-2020, for all the industries listed in the dataset and this decrease in working hours varies across the industries.


```python
#appending the newly created feature in the dataframe
x=pd.Series(x,name='Increased/Decreased Percentage',index=workhours_t.columns )
workhours_t=workhours_t.append(x,ignore_index=False)
workhours_t.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>North American Industry Classification System (NAICS)5</th>
      <th>Total actual hours worked, all industries 6</th>
      <th>Goods-producing sector 7</th>
      <th>Agriculture 8</th>
      <th>Forestry, fishing, mining, quarrying, oil and gas 9 10</th>
      <th>Utilities</th>
      <th>Construction</th>
      <th>Manufacturing</th>
      <th>Services-producing sector 11</th>
      <th>Wholesale and retail trade</th>
      <th>Transportation and warehousing</th>
      <th>Finance, insurance, real estate, rental and leasing</th>
      <th>Professional, scientific and technical services</th>
      <th>Business, building and other support services 12</th>
      <th>Educational services</th>
      <th>Health care and social assistance</th>
      <th>Information, culture and recreation</th>
      <th>Accommodation and food services</th>
      <th>Other services (except public administration)</th>
      <th>Public administration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Feb-20</th>
      <td>632365.700000</td>
      <td>150569.100000</td>
      <td>12208.400000</td>
      <td>12543.700000</td>
      <td>5217.200000</td>
      <td>55112.800000</td>
      <td>65487.000000</td>
      <td>481796.500000</td>
      <td>90184.100000</td>
      <td>37784.400000</td>
      <td>42511.400000</td>
      <td>53964.100000</td>
      <td>24081.500000</td>
      <td>39392.200000</td>
      <td>76261.900000</td>
      <td>23991.200000</td>
      <td>33505.200000</td>
      <td>26430.600000</td>
      <td>33689.900000</td>
    </tr>
    <tr>
      <th>Mar-20</th>
      <td>536710.100000</td>
      <td>138063.200000</td>
      <td>11622.500000</td>
      <td>12476.700000</td>
      <td>4928.800000</td>
      <td>48179.300000</td>
      <td>60855.900000</td>
      <td>398646.900000</td>
      <td>77315.600000</td>
      <td>33109.800000</td>
      <td>39668.600000</td>
      <td>49648.400000</td>
      <td>20834.100000</td>
      <td>28027.700000</td>
      <td>62618.600000</td>
      <td>16639.700000</td>
      <td>19739.200000</td>
      <td>20541.700000</td>
      <td>30503.700000</td>
    </tr>
    <tr>
      <th>Apr-20</th>
      <td>456983.800000</td>
      <td>104590.100000</td>
      <td>10908.500000</td>
      <td>10387.500000</td>
      <td>4764.700000</td>
      <td>32185.400000</td>
      <td>46343.900000</td>
      <td>352393.800000</td>
      <td>62226.500000</td>
      <td>27337.900000</td>
      <td>36854.500000</td>
      <td>47230.900000</td>
      <td>16783.700000</td>
      <td>30110.900000</td>
      <td>59254.400000</td>
      <td>15001.600000</td>
      <td>12120.800000</td>
      <td>13672.200000</td>
      <td>31800.500000</td>
    </tr>
    <tr>
      <th>May-20</th>
      <td>485951.200000</td>
      <td>116968.000000</td>
      <td>10995.900000</td>
      <td>11342.000000</td>
      <td>4922.400000</td>
      <td>38314.700000</td>
      <td>51393.000000</td>
      <td>368983.200000</td>
      <td>69101.300000</td>
      <td>27222.400000</td>
      <td>38328.400000</td>
      <td>47651.800000</td>
      <td>16624.100000</td>
      <td>32935.700000</td>
      <td>61796.000000</td>
      <td>15087.100000</td>
      <td>12408.800000</td>
      <td>15473.900000</td>
      <td>32353.600000</td>
    </tr>
    <tr>
      <th>Increased/Decreased Percentage</th>
      <td>-22.238651</td>
      <td>-20.712401</td>
      <td>-9.645267</td>
      <td>-13.296079</td>
      <td>-4.220419</td>
      <td>-28.534417</td>
      <td>-19.092389</td>
      <td>-22.710298</td>
      <td>-21.739657</td>
      <td>-27.569949</td>
      <td>-8.649031</td>
      <td>-13.027814</td>
      <td>-30.786557</td>
      <td>-16.871444</td>
      <td>-18.114009</td>
      <td>-34.489646</td>
      <td>-62.630737</td>
      <td>-40.173904</td>
      <td>-3.621177</td>
    </tr>
  </tbody>
</table>
</div>



# Analyzing using visualizations

*Visualizing the GDP by industry:*

Lets first visualize the GDP of the industries for the month of Nov-2019 and Mar-2020 to see how the contribution of the industries to the GDP changed over these months.


```python
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.bar(gdp_t.columns,height=gdp_t.loc['Nov-19'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('GDP')
plt.title(' GDP by industry in Nov-2019')

plt.subplot(1,2,2)
plt.bar(gdp_t.columns,height=gdp_t.loc['Mar-20'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('GDP')
plt.title(' GDP by industry in Mar-2020')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_53_0.png)


So as we can see, the GDP contribution by the industries has decreased which probably is the impact of the COVID-19 and the country-wide lockdown imposed for it. Though the GDP industry wise has decreased the share of their contribution compared to other industries have remained more or less the same. To find out which industries have the most and least decrease in their GDP, let's visualize the percentage increase/decrease in GDP industry wise.


```python
#plotting the decrease in GDP by industry
plt.figure(figsize=(15,10))
plt.bar(x=gdp_t.columns,height=-gdp_t.loc['Increased/Decreased Percentage'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('% Decrease in GDP')
plt.title(' Percentage decrease in GDP by industry ')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_55_0.png)


Thus from the above plot we can observe that the decrease in GDP was highest for the 'Arts, entertainment and recreation' industry (decreased by more than 41.28%), followed by the 'Accomodation and food services' industry (decreased by more than 37.67%), these are the hardest hit industries in terms of GDP. The GDP for the 'Real estate and rental and leasing' (decreased by 0.2%) and the 'Non-durable manfucaturing'(decreased by 0.083%) industries did not decrease much over the months from Nov-2019 to Mar-2020.

*GDP trends of industries over the period of Nov-2019 to Mar-2020:*


```python
#plotting some of the industry and their GDP trend
plt.figure(figsize=(15,10))
plt.plot(gdp_t.iloc[:-1,30:])
plt.xlabel('Months')
plt.ylabel('GDP')
plt.title('GDP trends for Nov-19 to Mar-20 industry wise')
plt.legend(gdp_t.iloc[:-1,28:].columns,loc='lower right')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_58_0.png)



```python
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(gdp_t['Arts, entertainment and recreation  [71]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Arts, entertainment and recreation')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,2)
plt.plot(gdp_t['Accommodation and food services  [72]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Accommodation and food services')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,3)
plt.plot(gdp_t['Non-durable manufacturing industries  [T011]4'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Non-durable manufacturing industries')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,4)
plt.plot(gdp_t['Real estate and rental and leasing  [53]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Real estate and rental and leasing')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.tight_layout()
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_59_0.png)


* Thus as we saw before and can also be noted from the above plots that the GDP for industries like 'Arts, entertainment and recreation', Accommodation and food services' have significantly decreased over these months.
* And industries like 'Non-durable manufacturing' and 'Real estate and rental and leasing' which were initially experiencing a steep increase in GDP and would have probably further increased but due to the COVID-19 situation faced a declining rate over the time.

*Visualizing the monthly CPI product/product group wise:*

Now lets visualize the CPI of the product/product groups for the month of Dec-2019 and Apr-2020 to see how the CPI of the products changed over these months.


```python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.bar(cpi_t.columns,height=cpi_t.loc['Dec-19'])
plt.xticks(rotation=90)
plt.xlabel('Product/Product Group')
plt.ylabel('CPI')
plt.title(' CPI by product for Dec-2019 ')


plt.subplot(1,2,2)
plt.bar(cpi_t.columns,height=cpi_t.loc['Apr-20'])
plt.xticks(rotation=90)
plt.xlabel('Product/Product Group')
plt.ylabel('CPI')
plt.title(' CPI by product for Apr-2020 ')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_63_0.png)


So as we can observe that the CPI for certain products have like 'Gasoline' and 'Energy' have significantly decreased which is probably due to the lock-down imposed during these months. To better analyze the change in CPI product wise, let's visualize the percentage change in CPI over these months:


```python
#plotting the change in CPY by product/product group
plt.figure(figsize=(15,10))
plt.bar(x=cpi_t.columns,height=cpi_t.loc['Increased/Decreased Percentage'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('% Change in CPI')
plt.title(' Percentage Change in CPI by product/product groups ')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_65_0.png)


* Thus from the above plot we can observe that the CPI decreased for some products and there was a significant decrease in CPI for 'Gasoline'(decreased by more than 31.48% ) and 'Energy'(decreased by more than 18.95% ). 
* Whereas the CPI increased for most of the products like 'Food'(inncreased by 1.38%) and 'Recreation, education and reading' (increased by 1.59%).

*CPI trends of products over the months of Dec-2019 to Apr-2020:*


```python
plt.figure(figsize=(15,10))
plt.plot(cpi_t.iloc[:-1,2:])
plt.xlabel('Months')
plt.ylabel('CPI')
plt.legend(cpi_t.iloc[:,2:].columns,loc='lower left')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_68_0.png)



```python
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(cpi_t['Gasoline'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])
plt.xlabel('Months')
plt.ylabel('CPI')
plt.title('CPI trend for product : Gasoline')

plt.subplot(2,2,2)
plt.plot(cpi_t['Energy 7'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])
plt.xlabel('Months')
plt.ylabel('CPI')
plt.title('CPI trend for product : Energy')

plt.subplot(2,2,3)
plt.plot(cpi_t['Food 5'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])
plt.xlabel('Months')
plt.ylabel('CPI')
plt.title('CPI trend for product : Food')

plt.subplot(2,2,4)
plt.plot(cpi_t['Recreation, education and reading'].loc[['Dec-19','Jan-20','Feb-20','Mar-20','Apr-20']])
plt.xlabel('Months')
plt.ylabel('CPI')
plt.title('CPI trend for product : Recreation, education and reading')

plt.tight_layout()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_69_0.png)


* Thus as we saw before and can also note from the above plots that the CPI for products like 'Gasoline', 'Energy' have significantly decreased over these months.
* And products like 'Food' and 'Recreation, education and reading' has experienced an increase over the time. Though the CPI for 'Recreation, education and reading' seems to be decreasing now but the CPI for 'Food' is still increasing steeply

*Visualizing the working hours by industries:*

Lets now visualize the working hours industry wise for the month of Jan-2020 and May-2020 to see how the working hours of the industries changed over these months:


```python
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.bar(workhours_t.columns,height=workhours_t.loc['Jan-20'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('Workhours')
plt.title(' Workhours by industry for Jan-2020 ')


plt.subplot(1,2,2)
plt.bar(workhours_t.columns,height=workhours_t.loc['May-20'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('Workhours')
plt.title(' Workhours by industry for May-2020 ')
plt.show()
plt.tight_layout()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_73_0.png)



    <Figure size 432x288 with 0 Axes>


So as we can observe that the workhours for all the listed industries have decreased. To better analyze the decreas in the workhours industry wise, let's visualize the percentage decrease in workhours over these months:


```python
plt.figure(figsize=(15,10))
plt.bar(x=workhours_t.columns,height=-workhours_t.loc['Increased/Decreased Percentage'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('% Decrease in Workhours')
plt.title('Percentage decrease in workhours industry wise')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_75_0.png)


* Thus from the above plot we can observe that the decrease in workhours was highest for the 'Accomodation and food services' industry (decreased by more than 62.63%), followed by the 'Other services (except public administration)' industry (decreased by more than 40.17%), these are the hardest hit industries in terms of workhours. 
* The workhours for the 'Public Administration' (decreased by 3.62%) and the 'Utilities'(decreased by 4.22%) industries did not decrease much over the months from Jan-2020 to May-2020.

*Workhours trend industry wise over the period of Jan-May 2020:*


```python
plt.figure(figsize=(15,10))
plt.plot(workhours_t.iloc[:-1,1:19])
plt.xlabel('Months')
plt.ylabel('Hours')
plt.title('Workhours trend for Jan-20 to May-20 industry wise')
plt.legend(workhours_t.iloc[:,1:].columns,loc='lower right')
plt.show()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_78_0.png)



```python
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
plt.plot(workhours_t['Professional, scientific and technical services'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])
plt.xlabel('Months')
plt.ylabel('Hours')
plt.title('Workhours trend for Professional, scientific and technical services industry')

plt.subplot(2,2,2)
plt.plot(workhours_t['Services-producing sector 11'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])
plt.xlabel('Months')
plt.ylabel('Hours')
plt.title('Workhours trend for Services-producing sector')

plt.subplot(2,2,3)
plt.plot(workhours_t['Accommodation and food services'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])
plt.xlabel('Months')
plt.ylabel('Hours')
plt.title('Workhours trend for Accommodation and food services')

plt.subplot(2,2,4)
plt.plot(workhours_t['Public administration'].loc[['Jan-20','Feb-20','Mar-20','Apr-20','May-20']])
plt.xlabel('Months')
plt.ylabel('Hours')
plt.title('Workhours trend for Public administration')

plt.tight_layout()
```


![png](/images/Canada_Covid19_Impact_Analysis_files/Canada_Covid19_Impact_Analysis_79_0.png)


* Thus as we saw before and can also note from the above plots that the workhours for industries like 'Accomodation and food services' have significantly decreased over these months.
* And products like 'Public administration' and 'Servicing producing sector' has experienced a steep decrease in workhours during the initial phase of this period but now they are slowly increasing over the time and this change can be contributed to the slow reopening of the economy. 
