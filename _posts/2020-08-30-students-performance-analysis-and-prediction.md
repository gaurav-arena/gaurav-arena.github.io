---
title: "Analyzing Students Performance"
date: 2020-08-30
tags: [data wrangling, data science, data analyzing, predictive modeling]
header:
  image: "/images/study.jpg"
excerpt: "Data Wrangling, Data Science, Data Analyzing, Predictive Modeling"
mathjax: "true"
---

**Goal:** Firstly we will try to analyze the 'StudentsPerformance.csv' dataset to draw inference about whether factors like gender of the student, the race/ethnicity of the student, the level of education of their parents, the type of lunch they ate and the completition of the test preparation course has any impact on the scores obtained by the student in the tests.
Some of the questions that this analysis will try to answer are:

1. Does the gender of student plays a role in how they perform in various courses.
2. Does the educational background of the parents impact the students performance.
3. Does the ethnicity of the student has an impact on their performance.
4. Is completing the Test Preparation course help the students in performing better.
5. Does the quality of lunch the students consume leaves an impact on how they perform.

Finally, based on the analysis a prediction model will be trained to predict how the students will perform given the factors influencing their performance and will also evaluate the performance of the model. A more detailed step-wise analysis of the same project can be found [here](https://github.com/gaurav-arena/Students-Performance-Analysis-and-Prediction) 


## Exploring the 'Students Performance' dataset. 


```python
#reading the StudentsPerformance.csv file and viewing it
student = pd.read_csv("StudentsPerformance.csv")
student.head()
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
    </tr>
  </tbody>
</table>
</div>



As can be seen from the first five rows of the dataset, the dataset contains columns like 'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'math score', 'reading score',	'writing score'


To obtain the general information about the dataset with informations like the column names their data types and the count of non-null values for every column:

```python
student.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1000 entries, 0 to 999
    Data columns (total 8 columns):
     #   Column                       Non-Null Count  Dtype 
    ---  ------                       --------------  ----- 
     0   gender                       1000 non-null   object
     1   race/ethnicity               1000 non-null   object
     2   parental level of education  1000 non-null   object
     3   lunch                        1000 non-null   object
     4   test preparation course      1000 non-null   object
     5   math score                   1000 non-null   int64 
     6   reading score                1000 non-null   int64 
     7   writing score                1000 non-null   int64 
    dtypes: int64(3), object(5)
    memory usage: 62.6+ KB
    

So we can see that there are no null values in any column of the dataset, this makes the analysis easier.

To know the number of categories present in each categorical variable we need to analyze the categorical variables seperately:


```python
student.select_dtypes('object').nunique()
```




    gender                         2
    race/ethnicity                 5
    parental level of education    6
    lunch                          2
    test preparation course        2
    dtype: int64



Now since we know the number of unique categories present in each of the categorical variable, it is important to see what are these unique category values in each of them.


```python
print("Categories in 'gender' variable: ",end=" ")
print(student['gender'].unique())
print("Categories in 'race/ethnicity' variable: ",end=" ")
print(student['race/ethnicity'].unique())
print("Categories in 'parental level of education' variable: ",end=" ")
print(student['parental level of education'].unique())
print("Categories in 'lunch' variable: ",end=" ")
print(student['lunch'].unique())
print("Categories in 'test preparation course' variable: ",end=" ")
print(student['test preparation course'].unique())
```

    Categories in 'gender' variable:  ['female' 'male']
    Categories in 'race/ethnicity' variable:  ['group B' 'group C' 'group A' 'group D' 'group E']
    Categories in 'parental level of education' variable:  ["bachelor's degree" 'some college' "master's degree" "associate's degree"
     'high school' 'some high school']
    Categories in 'lunch' variable:  ['standard' 'free/reduced']
    Categories in 'test preparation course' variable:  ['none' 'completed']
    

Now we need to analyze the quantitive/numerical columns in order to gain more them (informations like their count, their mean, their standard deviation, their minimum and maximum values) 
 
```python

student.describe()
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
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000.00000</td>
      <td>1000.000000</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>66.08900</td>
      <td>69.169000</td>
      <td>68.054000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15.16308</td>
      <td>14.600192</td>
      <td>15.195657</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>17.000000</td>
      <td>10.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>57.00000</td>
      <td>59.000000</td>
      <td>57.750000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>66.00000</td>
      <td>70.000000</td>
      <td>69.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>77.00000</td>
      <td>79.000000</td>
      <td>79.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>100.00000</td>
      <td>100.000000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>


## Feature Engineering
We will add a new feature(column) called 'Total Score' which will be basically the sum of the scores obtained in maths, writing and reading for every student. This feature will help in better analysing the overall performance of a student.


```python
#Total score = math score + reading score + writing score
student['Total Score']=student['math score']+student['reading score']+student['writing score']
```

We will also add a new column 'Pass/Fail', which will basically indicate the status of the student i.e. whether they have passed(P) or failed(F). To decide whether a student have passed we are evaluatin a condition on the total score obtained by the student. We are assuming that the passing criterion if a student has a Total Score of 120 or above then they Pass, otherwise, they Fail.


```python
#Criterion for getting a passing grade
def result(TS,MS,WS,RS ):
    if(TS>120 and MS>40 and WS>40 and RS>40):
        return 'P'
    else:
        return 'F'
    
```


Let's check the dataset again with the newly added two columns 'Total Score' & 'Pass/Fail'


```python
student.head()
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>math score</th>
      <th>reading score</th>
      <th>writing score</th>
      <th>Total Score</th>
      <th>Pass/Fail</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>72</td>
      <td>72</td>
      <td>74</td>
      <td>218</td>
      <td>P</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>69</td>
      <td>90</td>
      <td>88</td>
      <td>247</td>
      <td>P</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>90</td>
      <td>95</td>
      <td>93</td>
      <td>278</td>
      <td>P</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>47</td>
      <td>57</td>
      <td>44</td>
      <td>148</td>
      <td>P</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>76</td>
      <td>78</td>
      <td>75</td>
      <td>229</td>
      <td>P</td>
    </tr>
  </tbody>
</table>
</div>



Now using the newly added 'Pass/Fail' column, we will count the number of students passed and failed according to the passing criterion:


```python
#Displays the number of students passed and failed according to the passing criterion
student['Pass/Fail'].value_counts()
```




    P    939
    F     61
    Name: Pass/Fail, dtype: int64



So according to the count, a total of 939 students have passed and 61 students have failed out of the 1000 students.

Now lets try to visualize the performace of the students, sometimes visualization can help in exploring any underlying trends/relationships in a better way:


```python
plt.pie(student['Pass/Fail'].value_counts(),labels=['Pass','Fail'],autopct='%1.1f%%')
plt.title('Percentage of students Passed/Failed')
```


![png](/images/students-performance/students-performance-analysis-and-prediction_27_1.png)



```python
sns.countplot(student['Pass/Fail'])
plt.title('Bar-plot representing the count of students passed/failed')
```



![png](/images/students-performance/students-performance-analysis-and-prediction_28_1.png)


As the dataset contains both male and female students, we will try to analyze the variation of performance across the gender of the student and will try to findout if one gender performed better than the other.

For that first let's find out the number of male and female students in the dataset:
```python
# this displays the number of male and female students in the class
student['gender'].value_counts()
```




    female    518
    male      482
    Name: gender, dtype: int64



So as we can see that out of the 1000 students in the dataset, 518 are female and 482 are male. Thus the ratio of male and female students are almost same which is good. Now we will try to findout how did the male & female students performed when compared to each other.


```python
#to find out the percentage of female students passed
print("Percentage of female students passed: {0:.2f}%"
    .format((student[(student['gender']=='female') & (student['Pass/Fail']=='P')].shape[0]/student[student['gender']=='female'].shape[0])*100))

#to find out the percentage of male students passed
print("Percentage of male students passed: {0:.2f}%"
    .format((student[(student['gender']=='male') & (student['Pass/Fail']=='P')].shape[0]/student[student['gender']=='male'].shape[0])*100))
```

    Percentage of female students passed: 92.86%
    Percentage of male students passed: 95.02%
    

Therefore from the above analysis we can observe that the male students have overall performed relatively better than the female students.


```python
sns.countplot(student['Pass/Fail'],hue = student['gender'])
plt.ylabel('Number of students')
```

![png](/images/students-performance/students-performance-analysis-and-prediction_34_1.png)


We can observe from the above count plot that there is a variation between how both the genders performed and we can see that the male students have performed overall better than the female students, next we will try to analyze the performance of the students in the three different subjects and their variation across the gender.


```python
fig,ax = plt.subplots(3,1, figsize = (5,10))
sns.barplot(x=student['gender'],y=student['math score'], ax=ax[0], linewidth=2.5)
sns.barplot(x=student['gender'],y=student['reading score'], ax=ax[1],linewidth=2.5)
sns.barplot(x=student['gender'],y=student['writing score'], ax=ax[2],linewidth=2.5)
plt.tight_layout()
```


![png](/images/students-performance/students-performance-analysis-and-prediction_36_0.png)


As can be seen from the above barplots that the male students have performed better in maths whereas the female students have relatively performed better than the male students in both reading and writing exams.


```python
fig,ax = plt.subplots(3,1, figsize = (5,10))
sns.boxplot(x=student['gender'],y=student['math score'],ax=ax[0])
sns.boxplot(x=student['gender'],y=student['reading score'],ax=ax[1])
sns.boxplot(x=student['gender'],y=student['writing score'],ax=ax[2])
plt.tight_layout()
```


![png](/images/students-performance/students-performance-analysis-and-prediction_38_0.png)


The boxplots represent the performance of the male students vs. the performance of the female students in the three courses separately. As can be seen from the medians and the number of outliers, it can be concluded that the female students performed relatively poorer than the male students in maths but they out-performed the male students in both reading and writing scores. Thus we can conclude that, in this case the performance of a student in a course varies with the gender.

Next, we will try to analyse whether the **ethnicity/race** of the student plays any role in their performance.


```python
#number of students belonging to each race/ethnic group
student['race/ethnicity'].value_counts()
```




    group C    319
    group D    262
    group B    190
    group E    140
    group A     89
    Name: race/ethnicity, dtype: int64



Thus we can see that out of the 1000 students, 319 are from race group C, 262 are from group D,190 are from group B, 140 from group E and 89 are from the race group A. Now we will try to analyse how the students from the different race/ethnic groups have performed compared to each other.


```python
#number of students passed across the race/ethnic groups
print("The number of students passed across various race/ethnic group : ")
print(student['race/ethnicity'].loc[student['Pass/Fail']=='P'].value_counts())
sns.countplot(student['race/ethnicity'].loc[student['Pass/Fail']=='P'])
plt.xticks(rotation = 45)
```

    The number of students passed across various race/ethnic group : 
    group C    298
    group D    249
    group B    176
    group E    134
    group A     82
    Name: race/ethnicity, dtype: int64
    


![png](/images/students-performance/students-performance-analysis-and-prediction_43_2.png)



```python
sns.countplot(student['race/ethnicity'],hue=student['Pass/Fail'])
plt.ylabel('Number of students')
```



![png](/images/students-performance/students-performance-analysis-and-prediction_44_1.png)



```python
#to find out the percentage of students passed with the race/ethnicity  as 'group A'
print("Percentage of students passed with the race/ethnicity  as 'group A': {0:.2f}%"
    .format((student[(student['race/ethnicity']=='group A') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group A'].shape[0])*100))

#to find out the percentage of students passed with the race/ethnicity  as 'group B'
print("Percentage of students passed with the race/ethnicity  as 'group B': {0:.2f}%"
    .format((student[(student['race/ethnicity']=='group B') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group B'].shape[0])*100))

#to find out the percentage of students passed with the race/ethnicity  as 'group C'
print("Percentage of students passed with the race/ethnicity  as 'group C': {0:.2f}%"
    .format((student[(student['race/ethnicity']=='group C') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group C'].shape[0])*100))

#to find out the percentage of students passed with the race/ethnicity  as 'group D'
print("Percentage of students passed with the race/ethnicity  as 'group D': {0:.2f}%"
    .format((student[(student['race/ethnicity']=='group D') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group D'].shape[0])*100))

#to find out the percentage of students passed with the race/ethnicity  as 'group E'
print("Percentage of students passed with the race/ethnicity  as 'group E': {0:.2f}%"
    .format((student[(student['race/ethnicity']=='group E') & (student['Pass/Fail']=='P')].shape[0]/student[student['race/ethnicity']=='group E'].shape[0])*100))

```

    Percentage of students passed with the race/ethnicity  as 'group A': 92.13%
    Percentage of students passed with the race/ethnicity  as 'group B': 92.63%
    Percentage of students passed with the race/ethnicity  as 'group C': 93.42%
    Percentage of students passed with the race/ethnicity  as 'group D': 95.04%
    Percentage of students passed with the race/ethnicity  as 'group E': 95.71%
    

Thus from the above analysis we can observe that the race/ethnicity group 'group E' has performed better than all other groups and the group 'group A' has performed poorer than any other groups. It can also be observed that the performance of students in race/ethinicity group gets better as we move 'group A' to 'group E'.


```python
fig, ax = plt.subplots(3,1, figsize=(8,12))
sns.boxplot(x=student['race/ethnicity'],y=student['math score'],ax=ax[0])
sns.boxplot(x=student['race/ethnicity'],y=student['reading score'],ax=ax[1])
sns.boxplot(x=student['race/ethnicity'],y=student['writing score'],ax=ax[2])
plt.tight_layout()
```


![png](/images/students-performance/students-performance-analysis-and-prediction_47_0.png)


Thus it can also be noted in the above box-plots that, 'group A' has a relatively poorer performance in all the three courses whereas in comparison 'group E' performs relatively better than the other groups.

Now we will try to find the impact of the **educational background of the parents** on the students performance:


```python
#number of students having parents with various edication level
student['parental level of education'].value_counts()
```




    some college          226
    associate's degree    222
    high school           196
    some high school      179
    bachelor's degree     118
    master's degree        59
    Name: parental level of education, dtype: int64



Thus among the 1000 students, 226 students have parents with 'some college' background, 222 with 'associate's degree',196 have 'high school' background, 179 have parents with 'some high school' background, 118 with 'bachelor's degree',59 with 'master's degree' background. Now we will try to analyze how the performance of the students vary depending on their parents educational background.

Now, the number of students passed for each parental level of education:
```python
#number of students passed across the parental levels of education 
print("The number of students passed across the different parental levels of education: ")
print(student['parental level of education'].loc[student['Pass/Fail']=='P'].value_counts())
sns.countplot(student['parental level of education'].loc[student['Pass/Fail']=='P'])
plt.xticks(rotation = 45)
```

    The number of students passed across the different parental levels of education: 
    some college          216
    associate's degree    212
    high school           178
    some high school      162
    bachelor's degree     114
    master's degree        57
    Name: parental level of education, dtype: int64
    


![png](/images/students-performance/students-performance-analysis-and-prediction_52_2.png)


Like before, the percentage of students passed across the different parental level of education are also calculated:


    Percentage of students passed with the parental level of education as 'some college': 95.58%
    Percentage of students passed with the parental level of education as 'associate's degree': 95.50%
    Percentage of students passed with the parental level of education as 'high school': 90.82%
    Percentage of students passed with the parental level of education as 'some high school': 90.50%
    Percentage of students passed with the parental level of education as 'bachelor's degree': 96.61%
    Percentage of students passed with the parental level of education as 'master's degree': 96.61%
    


```python
plt.figure(figsize= (10,8))
sns.countplot(student['parental level of education'],hue=student['Pass/Fail'])
plt.xticks(rotation=90)
plt.ylabel('Number of students')
```

![png](/images/students-performance/students-performance-analysis-and-prediction_54_1.png)



```python
plt.figure(figsize=(10,5))
plt.title("Total Score across parental level of education of students")
sns.barplot(x=student['parental level of education'],y=student['Total Score'])
```

![png](/images/students-performance/students-performance-analysis-and-prediction_55_1.png)


As can be observed from the above plot that there is some influence the parent's background have on the student's performance. As can be seen, that students having parents with master's degree performed better than other and students with parents having some high school level of education performed poorer than the other groups. 

Next we are going to see how the **quality of lunch** impacts the performance of the students:


```python
#number of students having 'standard' lunch vs. number of students having 'free/reduced' lunch
student['lunch'].value_counts()
```




    standard        645
    free/reduced    355
    Name: lunch, dtype: int64



Thus out of the 1000 students, 645 have a standard lunch and 355 have a free/reduced lunch. Now we will analyze how the type of lunch varies the performance of the students.

The number of students passed for the two types of lunch:
```python
#number of students passed across the type of lunch 
student['lunch'].loc[student['Pass/Fail']=='P'].value_counts()
```




    standard        629
    free/reduced    310
    Name: lunch, dtype: int64




```python
sns.countplot(student['lunch'],hue=student['Pass/Fail'])
```

![png](/images/students-performance/students-performance-analysis-and-prediction_61_1.png)


The percentage of students passed for the two different lunch types:
```python
#to find out the percentage of students passed with the lunch type as 'standard'
print("Percentage of students passed with the lunch type as 'standard': {0:.2f}%"
    .format((student[(student['lunch']=='standard') & (student['Pass/Fail']=='P')].shape[0]/student[student['lunch']=='standard'].shape[0])*100))

#to find out the percentage of students passed with the lunch type as 'free/reduced'
print("Percentage of students passed with the lunch type as 'free/reduced': {0:.2f}%"
    .format((student[(student['lunch']=="free/reduced") & (student['Pass/Fail']=='P')].shape[0]/student[student['lunch']=="free/reduced"].shape[0])*100))

```

    Percentage of students passed with the lunch type as 'standard': 97.52%
    Percentage of students passed with the lunch type as 'free/reduced': 87.32%
    


```python
plt.figure(figsize=(5,5))
plt.title("Total Score across the type of lunch of the students")
sns.barplot(x=student['lunch'],y=student['Total Score'],hue=student['gender'])
```

![png](/images/students-performance/students-performance-analysis-and-prediction_63_1.png)


So as we can observe from the above plot, the type of lunch has an impact on the scores of the students. The students with 'standard' lunch performed better than the student with 'free/reduced' lunch.

Now we are going to find out whether completing the **'Test Preparation Course'** helps the students in performing better or not:

The number of students who completed the 'Test preparation course' vs. the students who didn't complete the course:
```python
#number of students who completed the 'Test preparation course' vs. the students who didn't complete the course
student['test preparation course'].value_counts()
```




    none         642
    completed    358
    Name: test preparation course, dtype: int64



Thus out of the 1000 students, 642 students didn't complete the 'Test preparation course' and 358 students completed it.

The number of students passed across the status of completion of the test preparation course:
```python
#number of students passed across the status of completion of the test preparation course 
print("The number of students passed across the status of completion of the test preparation course:")
print(student['test preparation course'].loc[student['Pass/Fail']=='P'].value_counts())

```

    The number of students passed across the status of completion of the test preparation course:
    none         591
    completed    348
    Name: test preparation course, dtype: int64
    

The percentage of students passed across the two different status of completion of the test preparation course:
```python
#to find out the percentage of students passed with the test preparation course status as 'none'
print("Percentage of students passed with the test preparation course status as 'none': {0:.2f}%"
    .format((student[(student['test preparation course']=='none') & (student['Pass/Fail']=='P')].shape[0]/student[student['test preparation course']=='none'].shape[0])*100))

#to find out the percentage of students passed with the test preparation course status as 'completed'
print("Percentage of students passed with the test preparation course status as 'completed': {0:.2f}%"
    .format((student[(student['test preparation course']=="completed") & (student['Pass/Fail']=='P')].shape[0]/student[student['test preparation course']=="completed"].shape[0])*100))

```

    Percentage of students passed with the test preparation course status as 'none': 92.06%
    Percentage of students passed with the test preparation course status as 'completed': 97.21%
    


```python
plt.figure(figsize=(5,5))
sns.barplot(x=student['test preparation course'],y=student['Total Score'])
plt.title("Total Score across the status of test prep course")
plt.xlabel('Status of Test Prep Course')
```

![png](/images/students-performance/students-performance-analysis-and-prediction_70_1.png)


As can be noted that the test preparation course has an impact on the performance of the students, 97.21% of the students who completed the 'Test Preparation Course'passed whereas 92.06% of the students who didn't complete 'Test Preparation Course' passed.

Now we will try to find and observe whether there is any correlation between how the students performed in the various courses.


```python
fig, ax = plt.subplots(3,1, figsize=(8,12))
sns.regplot(x=student['reading score'],y=student['writing score'],ax = ax[0])
sns.regplot(x=student['reading score'],y=student['math score'],ax = ax[1])
sns.regplot(x=student['writing score'],y=student['math score'],ax=ax[2])
plt.tight_layout()
```


![png](/images/students-performance/students-performance-analysis-and-prediction_73_0.png)


As can be seen from the above plots that there is a strong correlation between the scores.
To visualize the correlation in a better way, we produce a heat-map:


```python
student[student.columns[5:]].corr()['Total Score'][:]
```




    math score       0.918746
    reading score    0.970331
    writing score    0.965667
    Total Score      1.000000
    Name: Total Score, dtype: float64




```python
sns.heatmap(student.corr(), cmap ="Reds")
plt.xticks(rotation=90)
```

![png](/images/students-performance/students-performance-analysis-and-prediction_76_1.png)


As can be observed from the above heat-map that there is a strong correlation between 'reading score' and 'writing score'. The 'math score' is also correlated with the 'reading score' and 'writing score'.

So as we have analyzed the impact of different features on the student's performance and we observed that factors like 'gender', 'race/ethinicty', 'lunch', 'test preparation course' and 'parental level of education' impacted the scores obtained by the students.

The key findings obtained from the analysis are:
1. The overall performance of both the genders is almost the same with male students performing slightly better than the female. Moreover, it was also observed that the female students performed relatively poorer than the male students in Maths but they out-performed the male students in both Reading and Writing. Thus the performance of a student in a course has some correlation with their gender. Therefore, the school authority can put more emphasis for ensuring that both the genders perform equally well in all the courses.
2. The race/ethnicity group 'group E' has performed better than all other groups and the group 'group A' has performed poorer than any other ethnic groups. It was also observed that the performance of students in race/ethinicity group gets better as we move 'group A' to 'group E'. Thus the performance of a student in a course has some correlation with their race/ethnicity. Therefore, the authority should put more emphasis on equalizing the development of the ethnic groups.
3. The students having parents with master's degree performed better than other and students with parents having some high school level of education performed poorer than the other groups. Thus there is some correlation between the performance of a student with their parents educational background. Therefore, there should be more emphasis on the importance of education. 
4. The students with 'standard' lunch performed better than the student with 'free/reduced' lunch. Thus the quality of lunch also impacts the performance of a student. Therefore,a standard lunch for all the students can help ensuring an overall better performance. 
5. It was observed that the test preparation course does not have a significant impact on the performance of the students, 97.21% of the students who completed the 'Test Preparation Course'passed whereas 92.06% of the students who didn't complete 'Test Preparation Course' passed. Therefore, the school authority should analyze the efficiency of this course.

## Prediction Modeling

So as we have perfomed the analysis and have found some important insights and trends from the dataset, now we will try to train a model as a next step. The model should be able to accurately predict the **'Pass/Fail'** status of students provided with the features impacting the score of the student.

As we saw during the analysis that the course scores were highly correlated to each other, thus keeping all of them for model fitting does not make sense and rather it can impact the performance of the model, therefore, we have only kept the 'Total Score' in the set of features.
```python
X=student[['gender','race/ethnicity','parental level of education','lunch','test preparation course','Total Score']]
X.head()
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
      <th>gender</th>
      <th>race/ethnicity</th>
      <th>parental level of education</th>
      <th>lunch</th>
      <th>test preparation course</th>
      <th>Total Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>group B</td>
      <td>bachelor's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>218</td>
    </tr>
    <tr>
      <th>1</th>
      <td>female</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>completed</td>
      <td>247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>female</td>
      <td>group B</td>
      <td>master's degree</td>
      <td>standard</td>
      <td>none</td>
      <td>278</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>group A</td>
      <td>associate's degree</td>
      <td>free/reduced</td>
      <td>none</td>
      <td>148</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>group C</td>
      <td>some college</td>
      <td>standard</td>
      <td>none</td>
      <td>229</td>
    </tr>
  </tbody>
</table>
</div>



As we know, in order to train a model with categorical variables, they must be first converted into a form which can be utilized for the model fitting purpose. We have used the One Hot Encoding technique to transform the categorical variables.


```python
X_category = student[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]
```


```python
# Applying one-hot encoding to each column with categorical data
oh = ce.OneHotEncoder(cols=X_category.columns,handle_unknown='ignore', return_df=True,use_cat_names=True)
```

```python
X = oh.fit_transform(X)

```

  
```python
#collecting the status
y=student['Pass/Fail']
y.head()
```




    0    P
    1    P
    2    P
    3    P
    4    P
    Name: Pass/Fail, dtype: object



As the output/dependent is also categorical therefore we need to encode them:
```python
lb=LabelEncoder()
y=lb.fit_transform(y)
```

Dividing the dataset into training and validation subsets:
```python
# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
```

Using a Random Forest model and fitting it with the training subset:
```python
predictor = RandomForestRegressor()
predictor.fit(X_train,y_train)
```




    RandomForestRegressor()



Model predicting using the validation subset:
```python
#model predicting
preds=predictor.predict(X_valid)#predictions made by the model
```


```python
preds= np.where(preds<0.79,0,1)
```


Now we will try to evaluate the performace of our model by calculating the Mean Absolute Error  value and the R2 value


```python
#Calculating the Mean Absolute Error value
mae(y_valid,preds)
```




    0.02



As we can see that the Mean Absolute Error value for the model prediction is 0.02 which is an indication that the model is performing very accurately. For better evaluation, we will perform cross validation and will try to obtain the Mean Absolute Error(MAE) value after cross validation:


```python
scores = -1 * cross_val_score(predictor, X, y,cv=5,scoring='neg_mean_absolute_error')
print("MAE scores:\n", scores)
```

    MAE scores:
     [0.04615 0.0387  0.0384  0.0194  0.02555]
    

The best way to visualize the performance of a prediction model is through a confusion matrix:
```python
from sklearn.metrics import confusion_matrix

# creating a confusion matrix
cm = confusion_matrix(y_valid, preds)

# printing the confusion matrix
plt.rcParams['figure.figsize'] = (8, 8)
sns.heatmap(cm, annot = True, cmap = 'Reds')
plt.show()
```


![png](/images/students-performance/students-performance-analysis-and-prediction_101_0.png)


Another logical way of evaluating the performance of a predictor model is through finding out it's Precision and Recall values.



```python
print('Precision: %.3f' % ps(y_valid, preds))
```

    Precision: 0.989
    


```python
print('Recall: %.3f' % rs(y_valid, preds))
```

    Recall: 0.989
    

Thus from the above Precision and Recall values we can conclude that the model is predicting the performance of the students very accurately.
