---
title: "Web Scraping Project"
date: 2020-05-28
tags: [web scraping, data science, data scraping]
header:
  image: "/images/webs.png"
excerpt: "Web Scraping, Data Science, Data Scraping"
mathjax: "true"
---
#  Web scraping different news websites for Covid-19 headlines. 

## Goal : To scrap news websites to extract relevant important data related to latest Covid-19 headlines, this can further be utilized to supplement any existing dataset or analysis on Covid-19 with the latest news updates on Covid-19.

This project is on web scraping. The data available on the websites are usually unstructured, web scraping helps collect these unstructured data and store them in a structured form. There are different ways to extract data from websites such as online services, APIs or by writing our own code. In this project, we will implement web scraping with python using the BeautifulSoup library. A more step-wise detailed notebook for this project can be found [here](https://github.com/gaurav-arena/News-Website-Scrapping-Crawling)

*Pre-requisites:* 1. Basic knowledge of Python and HTML.
                  2. Python  3 should be installed.
                  3. All the required python libraries should be installed.
                  
We are basically scraping three news websites which belong to three different news organizations (CNN, NBC, CNBC) for data related Covid-19 headlines. This additional data extracted by web scraping these news websites can be used for complementing any existing dataset related to Covid-19 to perform better analysis.

Firstly,importing all the necessary libraries:
```python
from datetime import date
from bs4 import BeautifulSoup
import requests
import spacy
import en_core_web_sm
import pandas as pd
```

Collecting all the news website urls:
```python
cnn_url= 'https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html'
nbc_url= "https://www.nbcnews.com/health/coronavirus"
cnbc_rss_url = "https://www.cnbc.com/id/10000108/device/rss/rss.html"    
```

So before scraping the website it's important to understand their URL formats:

- If we look at the CNN website URL, it has a date attached to it and they update only the date for each new day and the rest remains same and thus it is a dynamic HTML website.
- For the CNBC website, we are using the RSS feed from the CNBC website, the RSS feed is basically an XML file.
- The NBC news URL doesn't have any date and it's a simple HTML website.

Collecting the URLs, format of the page, tags(under which the headlines are present) and website names for their identification in their respective lists:

```python
urls = [cnn_url,nbc_url,cnbc_rss_url]
formats=['html.parser','html.parser','xml']
tags = ['h2','h2','description']
website = ['CNN','NBC','CNBC']
```

For better understanding we will first start by scrapping only the CNN website:


```python
#setting the date format according to the date format used in the cnn url
today = date.today()
d = today.strftime('%m-%d-%y')
print('date =',d)
```

    date = 07-22-20
    

Getting the html for the CNN website: 

```python
html = requests.get(cnn_url).text
```

Creating a soup object using the BeautifulSoup library:
```python
soup = BeautifulSoup(html)
print(soup.title)#printing the title
```

    <title data-rh="true">July 7 coronavirus news</title>
    

To get an idea about context of the headlines in the news we are using the named entity extraction from the Spacy library.

Loading the entity extraction module of Spacy library:
```python
nlp = en_core_web_sm.load()
```

Printing the headlines and the named entity types of the context talked about in the news headlines in the CNN website:
```python
for link in soup.find_all('h2'): #finding all the h2 html tags as the CNN website contains the headlines under this tag
    
    print("Headline : {}".format(link.text))
    for ent in nlp(link.text).ents:
        print("\tText : {}, Entry : {}".format(ent.text,ent.label_))
```
Headlines from the CNN website along with their entity types:

    Headline : What you need to know
    Headline : Study finds coronavirus associated with neurological complications
    Headline : Colombia extends coronavirus lockdown measures
    	Text : Colombia, Entry : GPE
    Headline : Washington state governor blames Southern states reopening early for late Covid-19 test results
    	Text : Washington, Entry : GPE
    	Text : Southern, Entry : NORP
    Headline : South Dakota governor says she tested negative for Covid-19 after Fourth of July event
    	Text : South Dakota, Entry : GPE
    	Text : Fourth of July, Entry : DATE
    Headline : Texas Republicans have no plans to cancel in-person convention in Houston 
    	Text : Texas, Entry : GPE
    	Text : Republicans, Entry : NORP
    	Text : Houston, Entry : GPE
    Headline : Columbia University will welcome back 60% of undergraduate students in the fall
    	Text : Columbia University, Entry : ORG
    	Text : 60%, Entry : PERCENT
    Headline : More than 45,000 new coronavirus cases reported in Brazil
    	Text : More than 45,000, Entry : CARDINAL
    	Text : Brazil, Entry : GPE
    Headline : Bars ordered to close again in Shelby County, Tennessee
    	Text : Shelby County, Entry : GPE
    	Text : Tennessee, Entry : GPE
    Headline : Texas Education Agency says parents have option to choose remote learning for their children
    	Text : Texas Education Agency, Entry : ORG
    

Now we will be collecting headlines for all the websites now.

Crawling through the required web pages through their urls and printing the headlines and named entities associated: 
```python
crawl_len = 0
for url in urls:
    print("Crawling webpage ...{}".format(url))
    response = requests.get(url)
    soup = BeautifulSoup(response.content,formats[crawl_len])
    
    for link in soup.find_all(tags[crawl_len]):
        
        if(len(link.text.split(" ")) > 4):
            print("Headline : {}".format(link.text))
            
            entities=[]
            for ent in nlp(link.text).ents:
                print("\tText : {}, Entity : {}".format(ent.text,ent.label_))           
                
                
    crawl_len=crawl_len+1
```
Headlines and associated named entities for the three different news websites:

    Crawling webpage ...https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html
    Headline : What you need to know
    Headline : Study finds coronavirus associated with neurological complications
    Headline : Colombia extends coronavirus lockdown measures
    	Text : Colombia, Entity : GPE
    Headline : Washington state governor blames Southern states reopening early for late Covid-19 test results
    	Text : Washington, Entity : GPE
    	Text : Southern, Entity : NORP
    Headline : South Dakota governor says she tested negative for Covid-19 after Fourth of July event
    	Text : South Dakota, Entity : GPE
    	Text : Fourth of July, Entity : DATE
    Headline : Texas Republicans have no plans to cancel in-person convention in Houston 
    	Text : Texas, Entity : GPE
    	Text : Republicans, Entity : NORP
    	Text : Houston, Entity : GPE
    Headline : Columbia University will welcome back 60% of undergraduate students in the fall
    	Text : Columbia University, Entity : ORG
    	Text : 60%, Entity : PERCENT
    Headline : More than 45,000 new coronavirus cases reported in Brazil
    	Text : More than 45,000, Entity : CARDINAL
    	Text : Brazil, Entity : GPE
    Headline : Bars ordered to close again in Shelby County, Tennessee
    	Text : Shelby County, Entity : GPE
    	Text : Tennessee, Entity : GPE
    Headline : Texas Education Agency says parents have option to choose remote learning for their children
    	Text : Texas Education Agency, Entity : ORG
    Crawling webpage ...https://www.nbcnews.com/health/coronavirus
    Headline : Trump says coronavirus crisis will probably 'get worse before it gets better'
    	Text : Trump, Entity : ORG
    Headline : U.S. says China backed hackers who targeted COVID-19 vaccine research
    	Text : U.S., Entity : GPE
    	Text : China, Entity : GPE
    Headline : Coronavirus a 'Category 5 emergency' for Florida's older population
    	Text : Florida, Entity : GPE
    Headline : For the first time in 30 years, Walmart will be closed on Thanksgiving
    	Text : first, Entity : ORDINAL
    	Text : 30 years, Entity : DATE
    	Text : Walmart, Entity : PERSON
    	Text : Thanksgiving, Entity : DATE
    Headline : Puerto Rico wanted tourists, but as coronavirus spikes, it has changed plans
    	Text : Puerto Rico, Entity : GPE
    Headline : Highway deaths spike for third-straight month as drivers take advantage of empty roads
    	Text : third-straight month, Entity : DATE
    Headline : What's driving the resurgence in COVID-19 deaths?
    Headline : California surpasses New York in confirmed coronavirus cases
    	Text : California, Entity : GPE
    	Text : New York, Entity : GPE
    Headline : Fans will have to wear masks at NFL games this season — if there is a season with a live audience
    	Text : NFL, Entity : ORG
    	Text : this season, Entity : DATE
    Headline : Real estate is a seller's market as sales soar by 21 percent — but renters worry they will be left behind
    	Text : 21 percent, Entity : PERCENT
    Headline : More Wells Fargo customers say the bank decided to pause their mortgage payments without asking
    	Text : Wells Fargo, Entity : ORG
    Headline : I got catcalled in a mask. Here's what it revealed about masculinity.
    Headline : I thought the grief of losing my husband was over. Coronavirus brought it back.
    	Text : Coronavirus, Entity : ORG
    Headline : Hoping people believe face masks work is doomed to fail in an anti-vaccine world
    Headline : Nobody told me running a business means having a backdoor Lysol wipe supplier
    	Text : Lysol, Entity : GPE
    Headline : The complicated balancing act of church, state and the coronavirus 
    Headline : Trump rants about fraud. But here's the secret to keeping voting by mail secure.
    Headline : Comedy is no coronavirus vaccine. But can it make living through a pandemic easier?
    Headline : Arguments over reclining are so yesterday. Here’s what the future of flying looks like.
    	Text : yesterday, Entity : DATE
    Headline : Does air conditioning spread the coronavirus?
    Headline : Trump blames testing for spike in COVID-19 cases. Experts disagree. 
    Headline : Is this the second wave of COVID-19 in the U.S.? Or are we still in the first?
    	Text : second, Entity : ORDINAL
    	Text : U.S., Entity : GPE
    	Text : first, Entity : ORDINAL
    Headline : Trump falsely claims coronavirus numbers are 'going down almost everywhere'
    Headline : What you need to know about coronavirus home-testing kits
    Headline : Doctors, nurses and hospital staff
    Headline : Small businesses and their employees
    Headline : Senior citizens and the elderly
    Headline : What cleaning products kill the coronavirus?
    Headline : Are at-home testing kits available?
    Headline : Can you catch COVID-19 twice? 
    Headline : Gyms are eager to reopen, but are they safe?
    Headline : How will coronavirus change the way we fly?
    Headline : Covid Chronicles, Vol. 6: Virus testing becomes a test of character
    	Text : Covid Chronicles, Entity : PERSON
    	Text : Vol, Entity : PERSON
    	Text : 6, Entity : CARDINAL
    	Text : Virus, Entity : PERSON
    Headline : It's been months since the first U.S. coronavirus death. Here's a look at some of the lives lost.
    	Text : months, Entity : DATE
    	Text : first, Entity : ORDINAL
    	Text : U.S., Entity : GPE
    Headline : Affairs, hoarders and hope: Read anonymous confessions in the time of the coronavirus
    Headline : Millions are now out of work. Many wonder if there will be a job to go back to.
    	Text : Millions, Entity : CARDINAL
    Headline : MSNBC Special Report: Testing and the Road to Reopening
    	Text : MSNBC, Entity : ORG
    Headline : 'Into The Red Zone': An NBC News NOW documentary 
    	Text : NBC News, Entity : ORG
   ..........

Crawling through the webpages through the urls and printing only the headlines:
```python
crawl_len=0
news_dict=[]
for url in urls:
    response = requests.get(url)
    soup = BeautifulSoup(response.content,formats[crawl_len])
    
    for link in soup.find_all(tags[crawl_len]):
        
        if(len(link.text.split(" ")) > 4):
            print("Headline : {}".format(link.text))
            
            entities=[]
            entities = [(ent.text,ent.label_) for ent in nlp(link.text).ents]
            
            news_dict.append({'website': website[crawl_len],'url': url, 'headline':link.text, 'entities':entities})
            
    
    crawl_len=crawl_len+1          
```
Headlines:

    Headline : What you need to know
    Headline : Study finds coronavirus associated with neurological complications
    Headline : Colombia extends coronavirus lockdown measures
    Headline : Washington state governor blames Southern states reopening early for late Covid-19 test results
    Headline : South Dakota governor says she tested negative for Covid-19 after Fourth of July event
    Headline : Texas Republicans have no plans to cancel in-person convention in Houston 
    Headline : Columbia University will welcome back 60% of undergraduate students in the fall
    Headline : More than 45,000 new coronavirus cases reported in Brazil
    Headline : Bars ordered to close again in Shelby County, Tennessee
    Headline : Texas Education Agency says parents have option to choose remote learning for their children
    Headline : Trump says coronavirus crisis will probably 'get worse before it gets better'
    Headline : U.S. says China backed hackers who targeted COVID-19 vaccine research
    Headline : Coronavirus a 'Category 5 emergency' for Florida's older population
    Headline : For the first time in 30 years, Walmart will be closed on Thanksgiving
    Headline : Puerto Rico wanted tourists, but as coronavirus spikes, it has changed plans
    Headline : Highway deaths spike for third-straight month as drivers take advantage of empty roads
    Headline : What's driving the resurgence in COVID-19 deaths?
    Headline : California surpasses New York in confirmed coronavirus cases
    Headline : Fans will have to wear masks at NFL games this season — if there is a season with a live audience
    Headline : Real estate is a seller's market as sales soar by 21 percent — but renters worry they will be left behind
    Headline : More Wells Fargo customers say the bank decided to pause their mortgage payments without asking
    Headline : I got catcalled in a mask. Here's what it revealed about masculinity.
    Headline : I thought the grief of losing my husband was over. Coronavirus brought it back.
    Headline : Hoping people believe face masks work is doomed to fail in an anti-vaccine world
    Headline : Nobody told me running a business means having a backdoor Lysol wipe supplier
    Headline : The complicated balancing act of church, state and the coronavirus 
    Headline : Trump rants about fraud. But here's the secret to keeping voting by mail secure.
    Headline : Comedy is no coronavirus vaccine. But can it make living through a pandemic easier?
    Headline : Arguments over reclining are so yesterday. Here’s what the future of flying looks like.
    Headline : Does air conditioning spread the coronavirus?
    Headline : Trump blames testing for spike in COVID-19 cases. Experts disagree. 
    Headline : Is this the second wave of COVID-19 in the U.S.? Or are we still in the first?
    Headline : Trump falsely claims coronavirus numbers are 'going down almost everywhere'
    Headline : What you need to know about coronavirus home-testing kits
    Headline : Doctors, nurses and hospital staff
    Headline : Small businesses and their employees
    Headline : Senior citizens and the elderly
    Headline : What cleaning products kill the coronavirus?
    Headline : Are at-home testing kits available?
    Headline : Can you catch COVID-19 twice? 
    Headline : Gyms are eager to reopen, but are they safe?
    Headline : How will coronavirus change the way we fly?
    Headline : Covid Chronicles, Vol. 6: Virus testing becomes a test of character
    Headline : It's been months since the first U.S. coronavirus death. Here's a look at some of the lives lost.
    Headline : Affairs, hoarders and hope: Read anonymous confessions in the time of the coronavirus
    Headline : Millions are now out of work. Many wonder if there will be a job to go back to.
    Headline : MSNBC Special Report: Testing and the Road to Reopening
    Headline : 'Into The Red Zone': An NBC News NOW documentary 
    Headline : Portrait of a neighborhood under lockdown
    Headline : Photos capture empty cities across the globe
    Headline : New Yorkers are giving nightly ovations to health workers. These are their portraits
    Headline : "Hopefully the approval process will go very quickly, and we think we have a winner there," President Donald Trump said Wednesday.
    Headline : California reported more than 12,800 coronavirus cases on Tuesday, the highest reported daily tally the state has recorded so far, Gov. Gavin Newsom said Wednesday.
    ......
    

The best way to collect and store this data for further analysis in python is by storing it in a pandas dataframe and thus we are storing the extracted news website data in a dataframe, this can be stored as a CSV or Excel file and then it can be combined with other related datasets for more extensive analysis :


```python
#collecting the data into a dataframe
news_df=pd.DataFrame(news_dict)
```


```python
#viewing the dataframe
pd.set_option('max_colwidth',800)

news_df.head(20)
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
      <th>website</th>
      <th>url</th>
      <th>headline</th>
      <th>entities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>What you need to know</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Study finds coronavirus associated with neurological complications</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Colombia extends coronavirus lockdown measures</td>
      <td>[(Colombia, GPE)]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Washington state governor blames Southern states reopening early for late Covid-19 test results</td>
      <td>[(Washington, GPE), (Southern, NORP)]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>South Dakota governor says she tested negative for Covid-19 after Fourth of July event</td>
      <td>[(South Dakota, GPE), (Fourth of July, DATE)]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Texas Republicans have no plans to cancel in-person convention in Houston</td>
      <td>[(Texas, GPE), (Republicans, NORP), (Houston, GPE)]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Columbia University will welcome back 60% of undergraduate students in the fall</td>
      <td>[(Columbia University, ORG), (60%, PERCENT)]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>More than 45,000 new coronavirus cases reported in Brazil</td>
      <td>[(More than 45,000, CARDINAL), (Brazil, GPE)]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Bars ordered to close again in Shelby County, Tennessee</td>
      <td>[(Shelby County, GPE), (Tennessee, GPE)]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CNN</td>
      <td>https://www.cnn.com/world/live-news/coronavirus-pandemic-07-07-20-intl/index.html</td>
      <td>Texas Education Agency says parents have option to choose remote learning for their children</td>
      <td>[(Texas Education Agency, ORG)]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Trump says coronavirus crisis will probably 'get worse before it gets better'</td>
      <td>[(Trump, ORG)]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>U.S. says China backed hackers who targeted COVID-19 vaccine research</td>
      <td>[(U.S., GPE), (China, GPE)]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Coronavirus a 'Category 5 emergency' for Florida's older population</td>
      <td>[(Florida, GPE)]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>For the first time in 30 years, Walmart will be closed on Thanksgiving</td>
      <td>[(first, ORDINAL), (30 years, DATE), (Walmart, PERSON), (Thanksgiving, DATE)]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Puerto Rico wanted tourists, but as coronavirus spikes, it has changed plans</td>
      <td>[(Puerto Rico, GPE)]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Highway deaths spike for third-straight month as drivers take advantage of empty roads</td>
      <td>[(third-straight month, DATE)]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>What's driving the resurgence in COVID-19 deaths?</td>
      <td>[]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>California surpasses New York in confirmed coronavirus cases</td>
      <td>[(California, GPE), (New York, GPE)]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Fans will have to wear masks at NFL games this season — if there is a season with a live audience</td>
      <td>[(NFL, ORG), (this season, DATE)]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NBC</td>
      <td>https://www.nbcnews.com/health/coronavirus</td>
      <td>Real estate is a seller's market as sales soar by 21 percent — but renters worry they will be left behind</td>
      <td>[(21 percent, PERCENT)]</td>
    </tr>
  </tbody>
</table>
</div>



Thus this above dataset can be used to compliment any existing dataset on Covid-19 for better analysis with latest updates from the news websites.
