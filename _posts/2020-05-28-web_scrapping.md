---
title: "Web Scraping Project"
date: 2020-05-28
tags: [web scraping, data science, data scraping]
header:
  image: "/images/web-scraping.png"
excerpt: "Web Scraping, Data Science, Data Scraping"
mathjax: "true"
---
#  Web scraping different news websites for Covid-19 headlines. 

## Goal : To scrap news websites to extract relevant important data related to latest Covid-19 headlines, this can further be utilized to supplement any existing dataset or analysis on Covid-19 with the latest news updates on Covid-19.

This project is on web scraping. The data available on the websites are usually unstructured, web scraping helps collect these unstructured data and store them in a structured form. There are different ways to extract data from websites such as online services, APIs or by writing our own code. In this project, we will implement web scraping with python using the BeautifulSoup library. A more step-wise detailed notebook for this project can be found [link]

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
    Headline : Portrait of a neighborhood under lockdown
    Headline : Photos capture empty cities across the globe
    Headline : New Yorkers are giving nightly ovations to health workers. These are their portraits
    	Text : New Yorkers, Entity : NORP
    Headline : In case you missed it
    Headline : These states have the most coronavirus cases. See the day-by-day breakdown.
    Headline : Coronavirus stress among hospital workers leads to 'recharge rooms'
    	Text : Coronavirus, Entity : ORG
    Headline : FDA recalls more hand sanitizers due to toxic chemical
    	Text : FDA, Entity : ORG
    Headline : As pro sports return, athletes are divided on wearing masks
    Headline : Virus surge in June preceded by May surge in Yelp entries for bars, restaurants
    	Text : June, Entity : DATE
    	Text : May, Entity : DATE
    	Text : Yelp, Entity : GPE
    Headline : Trump returns to the coronavirus spotlight
    Headline : Experts voice concern over COVID-19 vaccine trials excluding hardest-hit populations
    Headline : Businesses are passing along COVID fees. Consumers are split on whether to accept them.
    	Text : COVID, Entity : ORG
    Headline : CDC: Antibody tests show virus rates 10 times higher than reported
    	Text : CDC, Entity : ORG
    	Text : 10, Entity : CARDINAL
    Headline : Lack of mask mandate and testing delays 'an equation for disaster' in Florida, experts warn
    	Text : Florida, Entity : GPE
    Crawling webpage ...https://www.cnbc.com/id/10000108/device/rss/rss.html
    Headline : Latest news for health care. 
    Headline : Contract tracing is being done after positive test in office building where many senior advisers and top officials work.
    Headline : "Hopefully the approval process will go very quickly, and we think we have a winner there," President Donald Trump said Wednesday.
    	Text : Donald Trump, Entity : PERSON
    	Text : Wednesday, Entity : DATE
    Headline : California reported more than 12,800 coronavirus cases on Tuesday, the highest reported daily tally the state has recorded so far, Gov. Gavin Newsom said Wednesday.
    	Text : California, Entity : GPE
    	Text : more than 12,800, Entity : CARDINAL
    	Text : Tuesday, Entity : DATE
    	Text : daily, Entity : DATE
    	Text : Gavin Newsom, Entity : PERSON
    	Text : Wednesday, Entity : DATE
    Headline : Covering all things Covid. Physician, bioethicist and author, Dr. Zeke Emanuel, discusses the pandemic's impact on our health care system, the ethics of reopening schools, and just how long until things go back to normal.
    	Text : Zeke Emanuel, Entity : PERSON
    Headline : Billionaire Bill Gates on Wednesday denied conspiracy theories that accuse the tech mogul and philanthropist of wanting to use coronavirus vaccines to implant tracking devices in people.
    	Text : Bill Gates, Entity : PERSON
    	Text : Wednesday, Entity : DATE
    Headline : Dr. David Satcher, former director of the Centers for Disease Control of Prevention, said he would grade the nation's leadership amid the pandemic a C "at best."
    	Text : David Satcher, Entity : PERSON
    	Text : the Centers for Disease Control of Prevention, Entity : ORG
    Headline : HHS abruptly changed the way hospitals report coronavirus data to the federal government, tying compliance to supply of a key drug.
    	Text : HHS, Entity : ORG
    Headline : New York Gov. Andrew Cuomo said he was encouraged by President Donald Trump's support of masks on Tuesday after months of resistance.
    	Text : New York, Entity : GPE
    	Text : Andrew Cuomo, Entity : PERSON
    	Text : Donald Trump, Entity : PERSON
    	Text : Tuesday, Entity : DATE
    	Text : months, Entity : DATE
    Headline : "Right now we have close to 1,000 casualties a day so if we don't change that trajectory, you could do the math," the former FDA commissioner told CNBC on Wednesday.
    	Text : 1,000, Entity : CARDINAL
    	Text : FDA, Entity : ORG
    	Text : CNBC, Entity : ORG
    	Text : Wednesday, Entity : DATE
    Headline : Brazil has struggled to contain the spread of the coronavirus and more than 80,000 people have died. Here's what went wrong, according to locals.
    	Text : Brazil, Entity : GPE
    	Text : more than 80,000, Entity : CARDINAL
    Headline : FEMA Administrator Pete Gaynor also told lawmakers that the reliance on overseas suppliers for PPE a "national security issue."
    	Text : FEMA, Entity : ORG
    	Text : Pete Gaynor, Entity : PERSON
    Headline : Fauci also said U.S. health officials do not see "an end in sight" to the pandemic.
    	Text : Fauci, Entity : ORG
    	Text : U.S., Entity : GPE
    Headline : The House Committee on Homeland Security plans to question FEMA administrator Pete Gaynor about the Federal government's coronavirus response and its "various shortcomings."
    	Text : The House Committee on Homeland Security, Entity : ORG
    	Text : FEMA, Entity : ORG
    	Text : Pete Gaynor, Entity : PERSON
    Headline : The coronavirus has infected more than 14.98 million people globally as of Wednesday, killing at least 617,415 people. 
    	Text : more than 14.98 million, Entity : CARDINAL
    	Text : Wednesday, Entity : DATE
    	Text : at least 617,415, Entity : CARDINAL
    Headline : Dow futures pointed to a modest decline at Wednesday's open as investors digest two breaking stories.
    	Text : Wednesday, Entity : DATE
    	Text : two, Entity : CARDINAL
    Headline : A large number of small businesses that closed in March — when restrictions around social movements when into effect — are not going to reopen even when the situation improves, according to Raghuram Rajan.
    	Text : March, Entity : DATE
    	Text : Raghuram Rajan, Entity : PERSON
    Headline : Biopharmaceutical company Arcturus and Singapore's Duke-NUS said they will begin human dosing tests of their potential coronavirus vaccine as soon as possible.
    	Text : Singapore, Entity : GPE
    	Text : Duke-NUS, Entity : ORG
    Headline : Trump also acknowledged Tuesday that the pandemic will probably "get worse before it gets better."
    	Text : Trump, Entity : ORG
    	Text : Tuesday, Entity : DATE
    Headline : "We need all states to ensure we're doing everything we can to better control the virus. If we can do that, then we'll be able to have the tests that we need," LabCorp CEO Adam Schechter told CNBC.
    	Text : Adam Schechter, Entity : PERSON
    	Text : CNBC, Entity : ORG
    Headline : President Donald Trump socialized years ago with accused child sex trafficker Jeffrey Epstein and his accused procurer, Ghislaine Maxwell.
    	Text : Donald Trump, Entity : PERSON
    	Text : years ago, Entity : DATE
    	Text : Jeffrey Epstein, Entity : PERSON
    	Text : Ghislaine Maxwell, Entity : PERSON
    Headline : Trump's comments come after the president has sent mixed messages for months on whether he supports the use of masks as a public health intervention to prevent the spread of the virus.
    	Text : Trump, Entity : ORG
    	Text : months, Entity : DATE
    Headline : After the coronavirus created a tidal wave of business interruption lawsuits, the insurance industry is pushing for a partly government-backed fund to cover future pandemics.
    Headline : President Donald Trump warned Tuesday the coronavirus pandemic in the United States will probably "get worse before it gets better."
    	Text : Donald Trump, Entity : PERSON
    	Text : Tuesday, Entity : DATE
    	Text : the United States, Entity : GPE
    Headline : Atlantic Health System CEO Brian Gragnolati told CNBC that mortality rates and length of stays in ICUs for coronavirus patients have come down as a result of new approaches to care.
    	Text : Atlantic Health System, Entity : ORG
    	Text : Brian Gragnolati, Entity : PERSON
    	Text : CNBC, Entity : ORG
    Headline : Texas and Florida hit a new grim record Monday for daily coronavirus deaths based on a seven-day moving average, as hospitalizations continue to surge in 34 states across the United States.
    	Text : Texas, Entity : GPE
    	Text : Florida, Entity : GPE
    	Text : Monday, Entity : DATE
    	Text : daily, Entity : DATE
    	Text : seven-day, Entity : DATE
    	Text : 34, Entity : CARDINAL
    	Text : the United States, Entity : GPE
    Headline : The White House said Trump, who faces a tough reelection fight, plans to use the restarted briefings to deliver "good news" about his response to the coronavirus pandemic.
    	Text : The White House, Entity : ORG
    	Text : Trump, Entity : ORG
    Headline : Fauci has faced criticism in recent weeks from President Donald Trump and other administration officials surrounding his response to the pandemic.
    	Text : Fauci, Entity : ORG
    	Text : recent weeks, Entity : DATE
    	Text : Donald Trump, Entity : PERSON
    Headline : United and other airlines are scrambling to cut costs as demand for air travel remains depressed in the coronavirus pandemic.
    	Text : United, Entity : ORG
    Headline : Last week, Cuomo announced that the state would place "enforcement teams" at New York airports to ensure compliance with the travel advisory.
    	Text : Last week, Entity : DATE
    	Text : Cuomo, Entity : PERSON
    	Text : New York, Entity : GPE
    Headline : "I'm sorry it's come to this, but it's a dangerous situation, and I've said it many, many times," Cuomo said during a press conference call.
    	Text : Cuomo, Entity : PERSON
    

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
    Headline : In case you missed it
    Headline : These states have the most coronavirus cases. See the day-by-day breakdown.
    Headline : Coronavirus stress among hospital workers leads to 'recharge rooms'
    Headline : FDA recalls more hand sanitizers due to toxic chemical
    Headline : As pro sports return, athletes are divided on wearing masks
    Headline : Virus surge in June preceded by May surge in Yelp entries for bars, restaurants
    Headline : Trump returns to the coronavirus spotlight
    Headline : Experts voice concern over COVID-19 vaccine trials excluding hardest-hit populations
    Headline : Businesses are passing along COVID fees. Consumers are split on whether to accept them.
    Headline : CDC: Antibody tests show virus rates 10 times higher than reported
    Headline : Lack of mask mandate and testing delays 'an equation for disaster' in Florida, experts warn
    Headline : Latest news for health care. 
    Headline : Contract tracing is being done after positive test in office building where many senior advisers and top officials work.
    Headline : "Hopefully the approval process will go very quickly, and we think we have a winner there," President Donald Trump said Wednesday.
    Headline : California reported more than 12,800 coronavirus cases on Tuesday, the highest reported daily tally the state has recorded so far, Gov. Gavin Newsom said Wednesday.
    Headline : Covering all things Covid. Physician, bioethicist and author, Dr. Zeke Emanuel, discusses the pandemic's impact on our health care system, the ethics of reopening schools, and just how long until things go back to normal.
    Headline : Billionaire Bill Gates on Wednesday denied conspiracy theories that accuse the tech mogul and philanthropist of wanting to use coronavirus vaccines to implant tracking devices in people.
    Headline : Dr. David Satcher, former director of the Centers for Disease Control of Prevention, said he would grade the nation's leadership amid the pandemic a C "at best."
    Headline : HHS abruptly changed the way hospitals report coronavirus data to the federal government, tying compliance to supply of a key drug.
    Headline : New York Gov. Andrew Cuomo said he was encouraged by President Donald Trump's support of masks on Tuesday after months of resistance.
    Headline : "Right now we have close to 1,000 casualties a day so if we don't change that trajectory, you could do the math," the former FDA commissioner told CNBC on Wednesday.
    Headline : Brazil has struggled to contain the spread of the coronavirus and more than 80,000 people have died. Here's what went wrong, according to locals.
    Headline : FEMA Administrator Pete Gaynor also told lawmakers that the reliance on overseas suppliers for PPE a "national security issue."
    Headline : Fauci also said U.S. health officials do not see "an end in sight" to the pandemic.
    Headline : The House Committee on Homeland Security plans to question FEMA administrator Pete Gaynor about the Federal government's coronavirus response and its "various shortcomings."
    Headline : The coronavirus has infected more than 14.98 million people globally as of Wednesday, killing at least 617,415 people. 
    Headline : Dow futures pointed to a modest decline at Wednesday's open as investors digest two breaking stories.
    Headline : A large number of small businesses that closed in March — when restrictions around social movements when into effect — are not going to reopen even when the situation improves, according to Raghuram Rajan.
    Headline : Biopharmaceutical company Arcturus and Singapore's Duke-NUS said they will begin human dosing tests of their potential coronavirus vaccine as soon as possible.
    Headline : Trump also acknowledged Tuesday that the pandemic will probably "get worse before it gets better."
    Headline : "We need all states to ensure we're doing everything we can to better control the virus. If we can do that, then we'll be able to have the tests that we need," LabCorp CEO Adam Schechter told CNBC.
    Headline : President Donald Trump socialized years ago with accused child sex trafficker Jeffrey Epstein and his accused procurer, Ghislaine Maxwell.
    Headline : Trump's comments come after the president has sent mixed messages for months on whether he supports the use of masks as a public health intervention to prevent the spread of the virus.
    Headline : After the coronavirus created a tidal wave of business interruption lawsuits, the insurance industry is pushing for a partly government-backed fund to cover future pandemics.
    Headline : President Donald Trump warned Tuesday the coronavirus pandemic in the United States will probably "get worse before it gets better."
    Headline : Atlantic Health System CEO Brian Gragnolati told CNBC that mortality rates and length of stays in ICUs for coronavirus patients have come down as a result of new approaches to care.
    Headline : Texas and Florida hit a new grim record Monday for daily coronavirus deaths based on a seven-day moving average, as hospitalizations continue to surge in 34 states across the United States.
    Headline : The White House said Trump, who faces a tough reelection fight, plans to use the restarted briefings to deliver "good news" about his response to the coronavirus pandemic.
    Headline : Fauci has faced criticism in recent weeks from President Donald Trump and other administration officials surrounding his response to the pandemic.
    Headline : United and other airlines are scrambling to cut costs as demand for air travel remains depressed in the coronavirus pandemic.
    Headline : Last week, Cuomo announced that the state would place "enforcement teams" at New York airports to ensure compliance with the travel advisory.
    Headline : "I'm sorry it's come to this, but it's a dangerous situation, and I've said it many, many times," Cuomo said during a press conference call.
    

The best way to collect and store this data for further analysis in python is by storing it in a pandas dataframe and thus we are storing the extracted news website data in a dataframe:


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
