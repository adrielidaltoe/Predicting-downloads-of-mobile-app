# Project 1 - Machine learning model to predict the downloads of mobile apps


# Setting the working directory
setwd("D:/DataScienceAcademy/FCD/Projetos/Projeto-Feedback1/Predicting-downloads-of-mobile-app")

# Packages for loading and manipulating data
library(data.table)
library(lubridate)
library(dplyr)
library(reshape)

# Packages for visualization
library(ggplot2)
library(gridExtra)
library(forcats)

# Categorical variables association
library(GoodmanKruskal)

# Packages for splitting the data and build machine learning models
library(randomForest)
library(class)
library(e1071)
library(caTools)
library(ROSE)
library(caret)
library(C50)
library(ROCR)

# Data balancing
library(DMwR)

# Importing customized functions
source("src/FuncTools.R")

# Importing the train data
train <- fread('train_sample.csv')
str(train)


# The train data contains 100000 rows and 8 columns.
dim(train)
View(train)

# Data description
# - ip: ip address of click.
# - app: app id for marketing.
# - device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# - os: os version id of user mobile phone
# - channel: channel id of mobile ad publisher
# - click_time: timestamp of click (UTC)
# - attributed_time: if user download the app after clicking an ad, this is the time of the app download
# - is_attributed: the target that is to be predicted, indicating the app was downloaded
# Note that ip, app, device, os, and channel are encoded.

# Features ip, app, device, os, channel and is_attibuted are categorical variables and will be
# properly transformed into categorical type in further manipulations.

## Searching for missing values
sapply(train, function(x){sum(is.na(x))})

# no missing values were detected, but there are rows with no values in column attributed_time
# marked with double quotes, that is why missing values for that column were not returned. At first, 
# this may not be a problem because in further manipulations this will be treated.


## Adding new features to be used in the exploratory data analysis
train <- train %>%
  mutate(datetime = as.POSIXct(click_time, format = "%Y-%m-%d %H:%M:%S"),
         date = as.Date(datetime),
         wday = wday(date),
         hour = hour(datetime),
         minute = lubridate::minute(datetime),
         attribute_POSIX = as.POSIXct(attributed_time, format = "%Y-%m-%d %H:%M:%S"),
         date_download = as.Date(attribute_POSIX),
         hour_download = hour(attribute_POSIX),
         ip_range = cut(ip, breaks = c(0,120000,210000,364757), labels = c(1,2,3)))
str(train)

################ Exploratory Data Analysis ###########################

## Summary of the data
summary(train)


## Number of unique values in each feature
cols = c('ip','app','device','os','channel','datetime','attribute_POSIX', 'date','is_attributed')
train %>% summarize_at(cols, n_distinct, na.rm = TRUE) %>%
  data.table::melt(variable.name = "features", value.name = "unique_values") %>%
  ggplot(aes(reorder(features, -unique_values), unique_values)) +
  geom_bar(stat = "identity", fill = "steelblue") + 
  scale_y_log10(breaks = c(50,100,250, 500, 10000, 50000)) +
  geom_text(aes(label = unique_values), vjust = 1.2, color = "white", size=3.5) +
  theme(axis.title.y=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank()) +
  ggtitle('Number of unique values in each feature') +
  labs(x = '', y = '')


## Exploring date column

# It is known from the former visualization that there are 4 unique values in date column, which means
# that data presented in train dataset is from four different days: 2017/11/06-09
unique(train$date)


## Exploring is_attributed column

# What is the amount of downloads?
table(train$is_attributed)
# 227 downloads from 100000 clicks.

# Visualizing the proportion between app downloads and clicks
# Pie chart
train %>% group_by(is_attributed) %>%
  summarise(n = n()) %>%
  mutate(is_attributed = ifelse(is_attributed == 0, 'Clique', 'Download')) %>%
  mutate(n=n*100/sum(n)) %>%
  mutate(ypos = cumsum(n)-0.5*n) %>%
  mutate(label = paste(n, '%', sep='')) %>%
  ggplot(aes(x="", y=n, fill=is_attributed)) +
  geom_bar(stat="identity", width=1, color="white") +
  coord_polar("y", start=0) +
  theme_void() + 
  geom_text(aes(y = ypos, label = label), color = "white")

# Note that only 0.227% of the clicks resulted in the download of the product.
# Moreover, the classes are imbalanced, which arises some problems in training ML algorithms, that are:
# Problems of performance (like accuracy) -> evaluation metrics guide the learning of the algorithm. The evaluation metrics
# do not take the minority class into account, then the predictions of these class will be compromised.
# The second problem is related with the lack of data for the minor class, which turns difficult to 
# construct accurate decisions boundaries between classes and uncover regularities within the small class. 
# The third problem is noise. Noisy data has a serius impact on minority classes than majority classes.
# Furthermore, standard ML algorithms tends to treat minority classes as noise. Thus, class imbalance will have to be treated
# before presenting the data to the algorithm.
names(train)

## What time did the downloads occur?
train %>% filter(hour_download != 'NA') %>%
  group_by(hour_download) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = as.factor(hour_download), y = n)) +
  geom_bar(stat = 'identity', fill='turquoise4', color="steelblue4") + 
  ggtitle('Distribution of app downloads throughout the day') + 
  xlab('Hour of the day') + 
  ylab('Downloads')

# Download number reduces between 17:00-22:00 and no downloads were observed at
# 16:00, 18:00 and 19:00.

# Distribution of clicks throughtout the 4 days (acumulate)
train %>% group_by(hour) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = as.factor(hour), y = n)) +
  geom_bar(stat = 'identity', fill = 'salmon1', color = 'salmon4') +
  ggtitle('Distribution of clicks throughtout the four days') +
  xlab('Hour') +
  ylab('Clicks')

# Mean clicks distribution per hour
train %>% 
  group_by(date, hour) %>%
  summarise(n = n()) %>%
  group_by(hour) %>%
  summarise(media = mean(n), std = sd(n)) %>%
  ggplot(aes(x= as.factor(hour), y= media)) + 
  geom_bar(stat="identity", fill = 'salmon1', color = 'salmon4',
           position=position_dodge()) +
  geom_errorbar(aes(ymin=media-std, ymax=media+std), width=.2,
                position=position_dodge(.9)) +
  xlab('Hour') +
  ylab('Mean clicks')

# It is observed that the downloads distribution followed the same trend as the distribution of clicks, i.e,
# reduces between 16:00-22:00. Moreover, the amount of clicks begin to increase around 23:00 and 
# fluctuate between 5000 and 6000 clicks in the interval 0:00 to 15:00.

# Next visualization shows the amount of clicks per minute.
train %>% group_by(date, minute) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = minute, y = n)) +
  geom_line(color = 'salmon1') +
  geom_point(color = 'salmon4') +
  facet_grid(. ~ date) +
  ggtitle('Clicks per minute') +
  xlab('Minute') +
  ylab('Clicks')

# And the amount of clicks per hour
train %>% group_by(date, hour) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = hour, y = n)) +
  geom_line(color = 'salmon1') +
  geom_point(color = 'salmon4') +
  facet_grid(. ~ date) +
  ggtitle('Clicks per hour') +
  xlab('Hour') +
  ylab('Clicks')

# Downloads per minute
train %>% filter(is_attributed == 1) %>%
  group_by(date, minute) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = minute, y = n)) +
  geom_line(color = 'salmon1') +
  geom_point(color = 'salmon4') +
  facet_grid(. ~ date) +
  ggtitle('Downloads per minute') +
  xlab('Minute') +
  ylab('Downloads')

# Downloads per hour
train %>% filter(is_attributed == 1) %>%
  group_by(date, hour_download) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = hour_download, y = n)) +
  geom_line(color = 'salmon1') +
  geom_point(color = 'salmon4') +
  facet_grid(. ~ date) +
  ggtitle('Downloads per hour') +
  xlab('Hour') +
  ylab('Downloads')

# Number of clicks and downloads between 16h - 22h.
train %>% filter(hour %in% seq(16,22)) %>%
  group_by(is_attributed) %>%
  summarise(total = n())


# The ad receives an average of 1667 clicks per minute in 4 days.
train %>% group_by(minute) %>%
  summarise(n = n()) %>%
  summarise(total = mean(n))
  

# Number of clicks per day.
train %>% group_by(date) %>%
  summarise(total = n()) %>%
  mutate(freq = total * 100 / sum(total)) %>%
  ggplot(aes(x = date, y = freq, fill = date)) +
  geom_bar(stat = 'identity', fill = 'tomato2', color = 'tomato4') +
  ggtitle('Percentage of clicks per day') +
  xlab('')+
  ylab('')

# It is observed that first day had approximately 90% less clicks compared to the other days.
# In this day, clicks were recorded from 16:00 to 21:59:50.
train %>% filter(date == '2017-11-06') %>%
  summarise(min = min(datetime), max = max(datetime))

# Let's see the distribution of clicks in each day.
ggplot(train, aes(x = hour, fill = date)) +
  geom_bar() + 
  facet_grid(. ~ date) + 
  ggtitle('Cliques por dia') +
  xlab('Hora') +
  ylab('') +
  theme(legend.position = "none")

# Now, the distribution of downloads in each day
train %>% filter(is_attributed == 1) %>%
  ggplot(aes(x = hour_download, fill = date_download)) +
  geom_bar() +
  facet_grid(. ~ date_download) + 
  ggtitle('Downloads por dia') +
  xlab('Hora') +
  ylab('') +
  theme(legend.position = "none")

# Number of downloads per day
train %>% filter(is_attributed == 1) %>%
  group_by(date_download) %>%
  summarise(downloads = n())
# 2017-11-06 presented only 2 downloads which makes sense, once the clicks in this day were between 16:00 to 22:00,
# hour of the day with lower clicks and lower downloads as well.
# Thus, it is noticeable that data is time dependent. 


## Exploring the IP attribute

# There are 34857 unique ip, which means that part of the clicks came from the same ip.
# How many IPs have more than 1 click? 

More_1_click <- train %>% group_by(ip) %>%
  summarise(n= n()) %>%
  filter(n > 1) %>%
  nrow()

Only_1_click <- train %>% group_by(ip) %>%
  summarise(n= n()) %>%
  filter(n == 1) %>%
  nrow()

rbind(Only_1_click,More_1_click)

# 17434 IPs have more than 1 click and 17423 IPs have only one click,
# i.e., half of the unique IPs has only 1 clique.


# Verifying if ips with more clicks presented downloads.

# 6 downloads came from 2 IPs; 221 downloads were made from different IPs.
train %>% filter(is_attributed == '1') %>%
  group_by(ip) %>%
  summarise(downloads_por_ip = n()) %>%
  group_by(downloads_por_ip) %>%
  summarise(number_of_occurrence = n())


# preparing ips with more than 1 click
df <- train %>% group_by(ip) %>%
  summarise(clicks = n()) %>%
  filter(clicks > 1)

# How many downloads came from IPs with more than 1 click?
joined_ip <- train %>% filter(is_attributed == '1') %>%
  inner_join(df, by = 'ip')
nrow(joined_ip)

# 77 downloads were made from IPs with more than 1 click
# And 7 downloads were made from IPs with more than 100 clicks, specifically from ips 5314, 5348, which
# presented more than 600 clicks. 3 downloads were made from each of those IPs.
joined_ip %>%
  filter(clicks > 100) %>%
  group_by(ip, date_download) %>%
  summarize(total = n())

# Thus, IP with higher amount of clicks does not necessarily have more downloads
train %>% filter(is_attributed == '1') %>%
  group_by(ip) %>%
  summarise(downloads = n()) %>%
  full_join(df, by = 'ip') %>%
  replace(is.na(.), 0) %>%
  arrange(desc(clicks)) %>%
  head(15) %>%
  mutate(IP = fct_reorder(as.factor(ip), desc(clicks))) %>%
  ggplot(aes(x = IP, y = clicks)) +
  geom_bar(stat = 'identity', color = 'orchid4', fill = 'orchid3') +
  geom_line(aes(x = IP, y = downloads*200, group = 1), size = 1, color="#00798C") +
  geom_point(aes(x = IP, y = downloads*200), color = '#00798C', size = 1.5) +
  scale_y_continuous(sec.axis = sec_axis(~./200, name = "Downloads")) +
  theme(axis.text.y.right = element_text(color = "#00798C")) +
  ggtitle('15 IPs com maior tráfego') +
  xlab('IP') +
  ylab('Cliques')

# Next visualization shows the distribuition of clicks and downloads throughout the IP numbers
p0 <- ggplot(train, aes(x = seq_along(ip), y = ip)) +
  geom_point(aes(col=factor(is_attributed)), alpha=0.8, size=0.05) +
  labs(x = 'index', y = 'IP')
p0

# It's observed that there are 3 ranges of ips with different concentrations of clicks: 0 - 120000; 120000 - 210000,
# 210000 - 360000 (approximated values). Clicks are more concentrated at the first range and decreases
# gradually in the second and third ranges. On the other hand, downloads are
# dispersed through all IPs, which is shown in the next visualization.
p1 <- train %>% filter(is_attributed == 1) %>%
  ggplot(aes(x = seq_along(ip), y = ip)) +
  geom_point(color = 'steelblue3') +
  labs(x = 'index', y = 'IP') +
  ggtitle('Distribuição dos downloads por IPs')

# Investigating the number of downloads in each ip range
p2 <- train %>% filter(is_attributed == 1) %>%
  group_by(ip_range) %>%
  count() %>%
  ggplot(aes(x = ip_range, y = n)) + 
  geom_bar(stat = 'identity', fill = 'steelblue4') +
  labs(x = 'IP range', y = 'Downloads') +
  ggtitle('Downloads por IP range')
grid.arrange(p1, p2, ncol = 2)

# Exploring the distribution of ips in each day
train %>% 
  ggplot(aes(x = seq_along(ip), y = ip)) +
  geom_point(color = 'slateblue3') +
  facet_grid(. ~ date) + 
  ggtitle('IP throughout the observed days') +
  xlab('Index') +
  ylab('IP') +
  theme(legend.position = "none")
# New ips are registered every day. As time goes by, the earlier range of ips keep getting more clicks,
# and new ip addresses appear.

# Next visualization shows the ip ranges that clicks come from in each hour. It is noticeable that first ip range
# predominates over the other ranges.
train %>%
  group_by(ip_range, date, hour) %>%
  count() %>%
  ggplot(aes(x = hour, y = n, fill = ip_range)) + 
  geom_bar(stat = 'identity') +
  facet_grid(.~ date) +
  labs(x = '', y = 'Clicks') +
  ggtitle('Distribution of IP range per hour')

# An interesting observation is that, despite less clicks, ip_range 3 presents more downloads than the others.
# Let's compare the number of clicks and the number of downloads for each ip-range.
p3 <- train %>% 
  group_by(ip_range) %>%
  count() %>%
  ggplot(aes(x = ip_range, y = n)) + 
  geom_bar(stat = 'identity', fill = 'steelblue4') +
  labs(x = 'IP range', y = 'Cliques') +
  ggtitle('Cliques por IP range')
grid.arrange(p3, p2, ncol = 2)

## IP with most clicks might be related with click fraud?

# collecting IPs with most clicks
ip_most_clicks <- train %>%
  group_by(ip) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  head(10) %>% 
  select(ip)

# Table summarizing the number of different devices and os accessing online ad through top 10 IPs
View(train %>%
  select(ip, device, os) %>%
  group_by(ip) %>%
  summarise(numero_de_modelos = length(unique(device)), numero_de_os = length(unique(os))) %>%
  filter(ip %in% ip_most_clicks$ip))

# Table summarizing the number of clicks coming from different ip, device, os pairs.
# "ip, device, os" pairs with a big amount of clicks have a higher chance to be click fraud.
View(train %>%
  select(ip, device, os, is_attributed) %>%
  group_by(ip, device, os, is_attributed) %>%
  summarise(numero_acessos = n()) %>%
  filter(ip %in% ip_most_clicks$ip) %>%
  arrange(desc(numero_acessos)))


## Exploring device attribute

# 25 of 100 devices had downloads. Device 0 and 1 had the most downloads.

# Next visualization presents a comparison between the devices that presented the most clicks and 
# the number of downloads with those devices. From the barplot we notice that around 94% of the clicks
# came from device 1 and 4.3% from device 2. Device 1 was also responsible for the most downloads (64%)
# followed by device 0 (0.5% of the clicks) with 23%.
d1 <- train %>% filter(is_attributed == 0) %>%
  group_by(device) %>%
  summarise(total = n()) %>%
  mutate(total = total*100/sum(total)) %>%
  mutate(factor = fct_reorder(as.factor(device), total)) %>%
  head(10) %>%
  ggplot(aes(x = total, y = factor)) +
  geom_bar(stat = 'identity', color = '#E7861B', fill = '#E7861B') +
  scale_x_continuous(breaks = seq(0, 100, by = 10)) +
  ggtitle('Cliques') +
  ylab('Device') +
  xlab('Cliques (%)')

d2 <- train %>% filter(is_attributed == 1) %>%
  group_by(device) %>%
  summarise(total = n()) %>%
  mutate(total = total*100/ sum(total))%>%
  mutate(factor = fct_reorder(as.factor(device), total)) %>%
  head(10) %>%
  ggplot(aes(x = total, y = factor)) +
  geom_bar(stat = 'identity', color = '#95A900', fill = '#95A900') +
  ggtitle('Downloads') +
  ylab('Device') +
  xlab('Downloads (%)')
grid.arrange(d1, d2, ncol = 2)


## Exploring OS attribute

# OS has 130 unique values. Next visualization shows the top 15 OS in clicks and downloads.
# Note that the top OS in downloads presents some OS that is not in the list of the
# top OS in clicks. But the OS with higher traffic is among those OS with higher frequency
# of downloads.

# Top 15 OS in clicks
os_click <- train %>% group_by(os) %>%
  summarise(n=n()) %>%
  arrange(desc(n))%>%
  mutate(os_clicks = n*100/nrow(train)) %>%
  head(15) %>%
  select(os, os_clicks)

# Top 15 OS in downloads
os_down <- train %>% filter(is_attributed == 1) %>%
  group_by(os) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  mutate(os_downloads = n*100/227) %>%
  head(15) %>%
  select(os, os_downloads)

# Clicks
o1 <- train %>% group_by(os) %>%
  summarise(n=n()) %>%
  arrange(desc(n))%>%
  mutate(os_clicks = n*100/sum(n)) %>%
  head(10) %>%
  mutate(os = fct_reorder(as.factor(os), os_clicks)) %>%
  ggplot(aes(x = os_clicks, y = os)) +
  geom_bar(stat = 'identity', color = '#00BA42', fill = '#00BA42') +
  ggtitle('Cliques') +
  ylab('os') +
  xlab('Cliques (%)')

# Downloads
o2 <- train %>% filter(is_attributed == 1) %>%
  group_by(os) %>%
  summarise(n = n()) %>%
  arrange(desc(n)) %>%
  mutate(os_downloads = n*100/sum(n)) %>%
  head(10) %>%
  mutate(os = fct_reorder(as.factor(os), os_downloads)) %>%
  ggplot(aes(x = os_downloads, y = os)) +
  geom_bar(stat = 'identity', color = '#00C08D', fill = '#00C08D') +
  ggtitle('Downloads') +
  ylab('os') +
  xlab('Downloads (%)')
grid.arrange(o1, o2, ncol = 2)



## Exploring app attribute

# 37 app id had downloads. The next visualization shows the top 10 apps in clicks.

# Top 10 app in clicks
app1<-train %>%
  group_by(app) %>%
  summarise(total = n()) %>%
  mutate(app_clicks = total*100/sum(total)) %>%
  select(app, app_clicks) %>%
  head(10) %>%
  mutate(app = fct_reorder(as.factor(app), app_clicks)) %>%
  ggplot(aes(x = app_clicks, y = app)) +
  geom_bar(stat = 'identity', color = '#F37B59', fill = '#F37B59') +
  ggtitle('Cliques') +
  ylab('app') +
  xlab('Cliques (%)')
  

# Apps with at least 2 downloads
app2<-train %>% 
  filter(is_attributed == 1) %>%
  group_by(app) %>%
  summarise(total = n()) %>%
  mutate(app_downloads = total*100/sum(total)) %>%
  select(app, app_downloads) %>%
  head(10) %>%
  mutate(app = fct_reorder(as.factor(app), app_downloads)) %>%
  ggplot(aes(x = app_downloads, y = app)) +
  geom_bar(stat = 'identity', color = '#ED8141', fill = '#ED8141') +
  ggtitle('Downloads') +
  ylab('app') +
  xlab('Downloads (%)')
grid.arrange(app1, app2, ncol = 2)


## Exploring channel attribute

# Channel has 161 unique values

# Let's look at the distribution of clicks in those channels. All 161 channels had clicks, next visualization
# shows that top 10 channels in clicks are not among the top 10 channels in downloads.

c1 <- train %>% group_by(channel) %>%
  summarise(channel_clicks = n()) %>%
  mutate(channel_clicks = channel_clicks*100/sum(channel_clicks)) %>%
  arrange(desc(channel_clicks)) %>%
  head(10) %>%
  ggplot(aes(x = channel_clicks, y = reorder(as.factor(channel), channel_clicks))) +
  geom_bar(stat = 'identity', color = 'wheat4', fill = 'thistle4') +
  ggtitle('Cliques') +
  xlab('Cliques (%)') +
  ylab('Channel')
  
c2 <- train %>%
  filter(is_attributed == 1) %>%
  group_by(channel) %>%
  summarise(channel_downloads = n()) %>%
  arrange(desc(channel_downloads)) %>%
  head(10) %>%
  ggplot(aes(x = channel_downloads, y = reorder(as.factor(channel), channel_downloads))) +
  geom_bar(stat = 'identity', color = 'wheat4', fill = 'violetred4') +
  ggtitle('Downloads') +
  xlab('Downloads (%)') +
  ylab('Channel')
grid.arrange(c1, c2, ncol = 2) 


# Once more, the channel with higher ammount of clicks wasn't the one with more downloads.

######### Data transformation and feature engineering ##############################

## Transforming the categorical variables into factors 
train_cat <- train %>%
  mutate(app = as.factor(app),
         device = as.factor(device),
         os = as.factor(os),
         channel = as.factor(channel),
         is_attributed = as.factor(is_attributed),
         wday = as.factor(wday),
         min_group = cut(minute, breaks = c(0,10,20,30,40,50,60), labels = c(1,2,3,4,5,6), include.lowest = TRUE)) %>%
  select(-c('click_time', 'attributed_time','attribute_POSIX', 'date_download','hour_download'))

### Adding new variables

## cont_time: is the time difference between a zero mark (specified 
# here as the index of min value in a datetime vector) and any other datetime value.
# The index is used here to perform the calculations between a specific time and the zero mark.
# The next variables count how many time the respective variables appeared together:
## ip_wday_hour
## ip_hour_app
## ip_hour_channel
## ip_hour_device
## ip_hour_os

train_cat <- train_cat %>%
  mutate(cont_time = time_count(train$datetime)) %>%
  add_count(ip, wday, hour, name = 'ip_wday_h') %>%
  add_count(ip, hour, channel, name = 'ip_h_channel') %>%
  add_count(ip, hour, os, name = 'ip_h_os') %>%
  add_count(ip, hour, app, name = 'ip_hour_app') %>%
  add_count(ip, hour, device, name = 'ip_h_dev') %>%
  select(-c('datetime'))

str(train_cat)
# The next functions group the categories of app, channel, os, device variables. group_categories() 
# mantain all the categories that have more than 10 downloads and group the rest. The new variables 
# created by this function are app_group, channel_group, os_group and device_group. group_target_prop(),
# instead of using the download absolut frequency, groups the categories according to the relative 
# frequency of downloads. It uses the minimum and maximum relative frequencies to divide the categories 
# into 5 groups. The variables created by this function are app_resp_group, channel_resp_group, 
# os_resp_group, device_resp_group.
names <- c('app','channel','os','device')
train_cat <- group_categories(train_cat, names, 'is_attributed')
train_cat <- group_target_prop(train_cat, names, 'is_attributed')

######### Studying the associations between variables #########

# I will apply some different statistics tests to evaluate the association between variables.
# Most of the variables in the original data are categorical variables, thus Pearson χ2 test
# will be used to verify the independency between variables. Contingency coefficient C, Cramer's V and 
# Thiel's U test will be applyied to investigate the association strengh.

# To study the association between categorical variables, a contigency table is used. 
# Contingency tables are used in statistics to summarize the relationship between 
# categorical variables. A contingency table is a special type of frequency distribution table,
# where two variables are shown simultaneously.

## Pearson χ2 test
# Statistical test applied to sets of categorical data to evaluate how likely it is that any 
# observed difference between the sets arise by chance.Three types of comparison can be extract
# from χ2 test: goodness of fit, homogeneity, and independence. 

# H0: frequency distribution is the same as the theoretical distribution (variables are independent)
  # If the test statistic exceeds the critical value of χ2 (p-value < 0.05), the null hypothesis can
  # be rejected, and the alternative hypothesis can be accepted.

# H1:  there is a difference between the distributions, i.e., variables are dependent of each other as well.
  # If the test statistic falls below the critical value of χ2 (p-value > 0.05), no clear conclusion
  # can be reached. The null hypothesis is sustained, but not necessarily accepted.

## Cramer's V
# Measures the association between two categorical variables. It is well known that this is biased estimator
# of their population counterparts and unfortunately, this bias may be rather large, even for large samples,
# potentially overestimating the true value (Reference: http://stats.lse.ac.uk/bergsma/pdf/cramerV3.pdf). Thus,
# care must be taken in evaluating the Cramer's V coefficient when contingency tables have different number 
# of columns and rows, once the dimension of the table influencies directly the value of this coefficient.
# If nrow != ncol, the coefficient will be biased to higher values. In summary, the larger the difference in 
# table's dimension, the larger the bias in Cramer's V, even when there is no association. Here
# https://www.coursera.org/lecture/inferential-statistics/2-03-interpreting-the-chi-squared-test-82xpS is 
# an excelent explanation about χ2 test and Cramer's V.
# Since the contingency tables of our data have different dimensions, cramerV method of rcompanion package,
# will be used, which accounts for an argument for bias.correction.

## The problem of symmetry
# Cramer's V (best choice When the contingency table is larger than 2 x 2) assumes symmetry in the data,
# f(x,y) = f(y,x). However, our data is asymmetric, once that you know the value of x (app, for example) not 
# necessarily you know the value of y (is_attributed). Theil's U and Goodman's and Kruskal's tau solves this
# limitation and calculates coefficients that are asymmetric with respect to the roles of X and Y. 

## Theil's U - uncertainty coefficient -> desctools package
# The function has interfaces for a table, a matrix, a data.frame and for single vectors. 
# direction: direction of the calculation. Can be row (default) or column, where row 
# calculates UncertCoef(R|C).

# Goodman's and Kruscak's tau is another asymmetric coefficient. There is a special case to be aware of in 
# tau: when K = N and the association tau(x,y) = 1, there are unique values in X related with 
# the values in Y.


# Evaluating the association between categorical variables 
names(train_cat)
str(train_cat)

var1 <- c('app', 'device','os','channel', 'wday', 'hour', 'min_group', 'ip_range', 'is_attributed')

var2 <- c('ip_wday_h','ip_h_channel','ip_h_os','ip_hour_app','ip_h_dev', 'is_attributed') 

var3 <- c('app_group', 'channel_group', 'device_group','os_group', 'is_attributed')

var4 <- c('app_resp_group', 'channel_resp_group', 'os_resp_group', 'device_resp_group', 'is_attributed')

features <- list(var1, var2, var3, var4)

for(var in features){
  categorical_association <- var_association(var, train_cat)
  plot.correlations(categorical_association)
  datagk = GKtauDataframe(train_cat[,..var])
  plot(datagk)
}

# The p-value from x2 test shows that is_attributed is independent of hour, wday, min_group, ip_wday_hour,
# ip_hour_app, ip_hour_channel, ip_hour_device, ip_hour_os, which is confirmed by the association 
# coefficients obtained with Theil's U, Cramer's V and Goodman's and Kruscak's tau.

# The correlation plot illustrates the asymmetry: is_attributed is somehow predicted if
# the value of some x variables is known, but much less information can be retrived by only knowing
# is_attributed. The association x -> y is stronger with variables app and channel, and weaker
# with device and os. Furthermore, channel and app are strongly associated; while device presented moderate
# association with add, channel and os. The lack of symmetry in this data is clearly shown by theil_sym 
# visualization, where only app and channel presented symmetry, i.e. app can be somewhat determined if 
# channel is known, and vice-versa. 

# Cramer's V coefficient gives no information about asymmetry of association, 
# however, the predicted association strengh given by Cramer's V for x -> y, being y is_attributed
# was somewhat comparable with those of Theil's U. On the other hand, Goodman and Kruskal tau coefficient
# demonstrated, in general, weaker associations.
# In summary, the associations are as follows:
# app -> is_attributed, channel -> is_attributed, os -> is_attributed, device -> is_attributed.
# app <-> channel (stronger association)
# app -> device, channel -> device, os -> device.

### Evaluating the dependence between cont_time and is_attributed
train_cat %>%
  ggplot(aes(x = cont_time)) +
  geom_histogram(bins = 70, color='steelblue4', fill = 'steelblue2', alpha=0.6, position = 'identity') +
  ggtitle('Distribution of click times counted in the interval of first and last click')

# It is noticed 4 different distributions related to the four days of downloads, similar to the distribution
# of total clicks per hour. Let's evaluate the boxplot of this variables for each category (is_attributed).
# There is no difference between cont_time between clicks that resulted in downloads and those that don't.

train_cat %>%
ggplot(aes(x = is_attributed, y = cont_time, fill = is_attributed)) +
  geom_boxplot(color= 'steelblue4', fill='steelblue2', alpha=0.8) +
  ggtitle('Boxplot of cont_time for each category of is_attributed')



######## Training machine learnig models ##################

## Dataframe to save the results of training
predictive_modeling_results <- as.data.frame(matrix(nrow = 0, ncol = 7))
names(predictive_modeling_results) <- c('model', 'ROC', 'Sens', 'Spec', 'ROCSD', 'SensSD', 'SpecSD')

## Dataframe to save the prediction results in test data
predictions <- as.data.frame(matrix(nrow = 0, ncol = 3))
names(predictions) <- c('model', 'roc_auc', 'prc_auc')


## Removing irrelevant features according to statistical tests. 
# The resp_group attributes will be removed because they have weaker association with is_attributed
names(train_cat)
train_model <- train_cat %>% select(c(app, device, os, channel, is_attributed, 
                                      ip_range, ip_h_dev, app_group, channel_group,
                                      os_group, device_group))

### Data split

str(train_model)
names(train_model)

set.seed(1045)
sample <- sample.split(train_model$is_attributed, SplitRatio = 0.80)

train_data <- subset(train_model, sample == TRUE)
test_data <- subset(train_model, sample == FALSE)

dim(train_data)
dim(test_data)
table(train_data$is_attributed)
table(test_data$is_attributed)


# Changing the labels of is_attributed to 0 == no and 1 == yes. This is a requirement of 
# caret package

train_data <- train_data %>% 
  mutate(is_attributed = factor(is_attributed, labels = c('no','yes')))

test_data <- test_data %>% 
  mutate(is_attributed = factor(is_attributed, labels = c('no','yes')))


#### Training ML models ######

# Parameters used during training
fitControl <- trainControl(method = "cv",
                           number = 5,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary,
                           verboseIter = TRUE)

########### Building a base-model ####################
grid <- expand.grid(.winnow = c(TRUE), .trials=c(1), .model="tree" )

set.seed(1045)
c50_base.model <- train(x = train_data[, -c('is_attributed')],
                        y = train_data$is_attributed,
                        metric = 'ROC',
                        tuneGrid = grid, 
                        trControl = fitControl, method="C5.0",verbose=FALSE)

# Adding the results to a dataframe
predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('c50_base.model')), c('model')), 
                                           c50_base.model$results[,-c(1:3)]))

## Saving the model 
saveRDS(c50_base.model, "models/c50_base.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(c50_base.model, test_data[,-c('is_attributed')],
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('c50_base.model')), c('model')), pred))



########### Balancing train_set ############

### Evaluating SMOTE perc.over parameter ### 

perc.ranges <- c(200,300,400,500,700,1000,1500,2000,3000,5000,10000,20000)

results = data.frame(matrix(nrow = 0, ncol = 4))
names(results) = c('perc.over', 'auc_cv', 'std', 'auc_test_data')

set.seed(1042)
for (i in perc.ranges){
  
  # balancing the data
  smoted_data <- SMOTE('is_attributed' ~., data = train_data, perc.over = i, k = 5,
                       perc.under = 200, set.seed = 1045)
  
  # Training the model with 5-fold cross validation
  fitControl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE,
                             summaryFunction = twoClassSummary,
                             verboseIter = TRUE)
  
  grid <- expand.grid(.winnow = c(TRUE), .trials=c(1), .model="tree" )
  
  c50model <- train(x = smoted_data[, -c('is_attributed')],
                    y = smoted_data$is_attributed,
                    metric = 'ROC',
                    tuneGrid = grid, 
                    trControl = fitControl, method="C5.0",verbose=FALSE)
  
  # Predictions on test_data
  pred<-prediction(predict(c50model, newdata = test_data[,-c('is_attributed')], type = 'prob')[, 2], 
                   labels =test_data$is_attributed, label.ordering = c('no', 'yes'))
  
  auc_ROCR <- performance(pred, measure = "auc")
  
  # Saving results in a dataframe
  results[(length(results$perc.over)+1), 1] <- i
  results[(length(results$perc.over)), 2] <- c50model$results$ROC
  results[(length(results$perc.over)), 3] <- c50model$results$ROCSD
  results[(length(results$perc.over)), 4] <- auc_ROCR@y.values[[1]]
}


# Saving smote tuning results 
balancing_tuning <- write.csv(results, 'balancing_tuning.csv', row.names = FALSE)

# Plot smote tuning results
colors <- c('cross-validation' = 'black', 'test on test_data' = 'blue')
results %>%
  ggplot(aes(x = perc.over)) +
  geom_line(aes(y = auc_cv, color = 'cross-validation')) +
  geom_point(aes(y = auc_cv, color = 'cross-validation')) +
  geom_errorbar(aes(ymin=auc_cv-std, ymax=auc_cv+std), width=.2,
                position=position_dodge(0.05)) +
  geom_line(aes(y = auc_test_data, color = 'test on test_data')) +
  geom_point(aes(y = auc_test_data, color = 'test on test_data')) + 
  ggtitle('Tuning SMOTE perc.over parameter') +
  scale_color_manual(values = colors)


################### Balanced data set ##################
smoted_data <- SMOTE('is_attributed' ~., data = train_data, perc.over = 20000, k = 5,
                     perc.under = 200, set.seed = 1045)

# Plot classes proportions
smoted_data %>% group_by(is_attributed) %>%
  summarise(n = n()) %>%
  mutate(is_attributed = ifelse(is_attributed == 'no', 'Clique', 'Download')) %>%
  mutate(n = n*100/sum(n)) %>%
  arrange((n)) %>%
  mutate(ypos = cumsum(n)-0.5*n) %>%
  mutate(label = paste(round(n,3), '%', sep='')) %>%
  ggplot(aes(x="", y=n, fill=is_attributed)) +
  geom_bar(stat="identity", width=1, color="white") +
  coord_polar("y", start=0) +
  theme_void() + 
  geom_text(aes(y = ypos, label = label), color = "white")

######## Training C5.0 model with balanced data #########################
grid <- expand.grid(.winnow = c(TRUE), .trials=c(1), .model="tree" )

set.seed(1045)
c50_base.smote <- train(x = smoted_data[, -c('is_attributed')],
                        y = smoted_data$is_attributed,
                        metric = 'ROC',
                        tuneGrid = grid, 
                        trControl = fitControl, method="C5.0",verbose=FALSE)

plot(varImp(c50_base.smote))
# Adding the results to training results dataframe
predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('c50_base.smote')), c('model')), 
                                           c50_base.smote$results[,-c(1:3)]))


## Saving the model 
saveRDS(c50_base.smote, "models/c50_base_smote.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(c50_base.smote, test_data[,-c('is_attributed')],
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('c50_base.smote')), c('model')), pred))



########### Training C5.0 model with balanced data and tunning parameters ####################
grid <- expand.grid(.winnow = c(TRUE, FALSE), .trials=c(1,3,6,9,12,14,20), .model="tree")

set.seed(1042)
c50model <- train(x = smoted_data[, -c('is_attributed')],
                  y = smoted_data$is_attributed,
                  metric = 'ROC',
                  tuneGrid = grid, 
                  trControl = fitControl, method="C5.0",verbose=FALSE)

# Plot of the cross-validation results
plot(c50model)
plot(varImp(c50model))

# Best tune: trials = 20, model = tree, winnow = TRUE 
c50model$bestTune

# saving the results of grid search with cross-validation
write.csv(x = c50model$results, file = 'c50tuning.csv', row.names = FALSE)
#c50_tuning <- read.csv('c50tuning.csv')

# Saving best performance in predictive_modeling_results dataframe
best_performance <- c50model$results%>%
  filter(ROC == max(ROC)) %>%
  select(-c(model, winnow, trials))

predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('c50model_balanced')), c('model')), 
                                           best_performance))

## Saving the model 
saveRDS(c50model, "models/c50model_smote_tuned.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(c50model, test_data[,-c('is_attributed')],
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('c50model_smote_tuned')), c('model')), pred))



############# Training Knn ########################
set.seed(1042)
knnFit <- train(x = smoted_data[, -c('is_attributed')],
                y = smoted_data$is_attributed, 
                method = "knn",
                metric = 'ROC',
                trControl = fitControl, 
                preProcess = c("center","scale"), 
                tuneLength = 10)


plot(knnFit)
plot(varImp(knnFit))

# Best tune: k = 17
knnFit$bestTune

# Saving the results of best performance
best_knn_perf <- knnFit$results %>%
  filter(ROC==max(ROC)) %>%
  select(-c(k))

predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('knn')), c('model')), 
                                           best_knn_perf))


## Saving the model 
saveRDS(knnFit, "models/knn.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(knnFit, test_data[,-c('is_attributed')],
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('kNN')), c('model')), pred))


##### Training Random Forest #####
# Random Forest can not handle categorical predictors with more than 53 categories. Thus,
# ip, app, device, os and channel attributes will be removed from the data.
names(smoted_data)
smoted_data_rf <- smoted_data[, -c(1:5)]
test_data_rf <- test_data[, -c(1:5)]

set.seed(1045)
rfFit <- train(x = smoted_data_rf,
               y = smoted_data$is_attributed, 
               method = "rf",
               metric = 'ROC',
               trControl = fitControl, 
               tuneLength = 10)

plot(varImp(rfFit))
plot(rfFit)

# Best tune: mtry = 2
rfFit$bestTune

# Saving the results of best performance
best_rf_perf <- rfFit$results %>%
  filter(ROC==max(ROC)) %>%
  select(-c(mtry))

# Saving best performance 
predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('Random Forest')), c('model')), 
                                           best_rf_perf))


## Saving the model 
saveRDS(rfFit, "models/rforest.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(rfFit, test_data_rf,
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('Random Forest')), c('model')), pred))



##### Training Gradient Boosting #####
library(gbm)

gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:10)*50, 
                        shrinkage = c(0.01,0.1,1),
                        n.minobsinnode = c(10,15,20))

set.seed(825)
gbmFit2 <- train(x = smoted_data[, -c('is_attributed')],
                 y = smoted_data$is_attributed, 
                 method = "gbm", 
                 trControl = fitControl, 
                 verbose = FALSE, 
                 metric = 'ROC',
                 tuneGrid = gbmGrid)

plot(gbmFit2)
plot(varImp(gbmFit2))

# Best tune: n.trees = 500, interaction.depth = 5, shrinkage = 0.1, n.minobsinnode = 20
gbmFit2$bestTune

# Saving the results of the best performance
gbm_rf_perf <- gbmFit2$results %>%
  filter(ROC==max(ROC)) %>%
  select(-c(shrinkage,interaction.depth,n.minobsinnode,n.trees))

predictive_modeling_results <- rbind(predictive_modeling_results, 
                                     cbind(setNames(as.data.frame(c('GBM')), c('model')), 
                                           gbm_rf_perf))

## Saving the model 
saveRDS(gbmFit2, "models/gbm.rds")


# Evaluating model c50_base.model with training data
pred <- as.data.frame(evaluate_model_prob(gbmFit2, test_data[,-c('is_attributed')],
                                          test_data$is_attributed))

# saving results in predictions dataframe
predictions <- rbind(predictions, 
                     cbind(setNames(as.data.frame(c('Gradient Boosting')), c('model')), pred))



### Saving results
write.csv(predictive_modeling_results, 'predictive_modeling_results.csv', row.names = FALSE)
write.csv(predictions, 'predictions.csv', row.names = FALSE)


## Prediction results
library(viridis)
library(hrbrthemes)

# Comparing C5.0 models
melt(predictions[c(1:3),], id.vars = 'model', measure.vars = c('roc_auc', 'prc_auc')) %>%
  ggplot(aes(x=model, y = value, fill = variable)) +
  geom_col(position="dodge") + 
  scale_fill_viridis(discrete = T) +
  ylim(0,1) +
  xlab('') +
  ylab('AUC') +
  ggtitle('Performance dos modelos C5.0')

# Comparing tuned models
melt(predictions[c(3:6),], id.vars = 'model', measure.vars = c('roc_auc', 'prc_auc')) %>%
  ggplot(aes(x=model, y = value, fill = variable)) +
  geom_col(position="dodge") + 
  scale_fill_viridis(discrete = T) +
  ylim(0,1) +
  xlab('')+
  ylab('AUC') +
  ggtitle('Comparação entre os modelos de diferentes algoritmos')


##### Comparing the performances of C5.0 and GBM models with new data

### Importing models
c50_smote_tuned <- readRDS('models/c50model_smote_tuned.rds')
gbmFit <- readRDS('models/gbm.rds')


# Predictions to plot ROC and PRC curves
predictions_c50_smote_tuned <- predict(c50_smote_tuned, newdata = test_data[,-c('is_attributed')], 
                                       type = 'prob')[,2]
predictions_gbm <- predict(gbmFit, newdata = test_data[,-c('is_attributed')], type = 'prob')[,2]


models_predictions <- prediction(list(predictions_c50_smote_tuned,
                                      predictions_gbm),
                                 labels =rep(list(test_data$is_attributed),2))

## Preparing ROC curves
rocs <- performance(models_predictions, "tpr", "fpr")

# Creating the legends
roc_legend <- c()
for(i in 1:(nrow(predictions)-1)){
  roc_legend <- c(roc_legend, paste(predictions$model[i+1],'AUC -',round(predictions$roc_auc[i+1],3)))
}


## Preparing Precision-Recall Curves
prc_curve <- performance(models_predictions, "prec", "rec")

# Creating the legends
prc_legend <- c()
for(i in 1:(nrow(predictions)-1)){
  prc_legend <- c(prc_legend, paste(predictions$model[i+1],'AUC -',round(predictions$prc_auc[i+1],3)))
}

prc_legend <- c(prc_legend, 'No Skill')


# Plotting the curves
par(mfrow = c(1, 2))

plot(rocs, col=as.list(1:2), lwd= 2, main="Curvas ROC",
     xlab = 'Taxa de Falso Positivos',
     ylab = 'Taxa de Verdadeiro Positivos')
legend(x="bottomright", bty = 'n',
       legend=roc_legend[c(2,5)], 
       fill=1:2)

plot(prc_curve, col=as.list(1:2),
     lwd= 2,
     main= "Curvas Precisão/Recall",
     xlab = "Taxa de Verdadeiro Positivos",
     ylab = 'Precisão')
abline(h = 0.0023, col="green", lty=2, lwd=3)
legend(x="topright", bty = 'n',
       legend= prc_legend[c(2,5,6)], fill = c(1:3))



# Creating a confusion matrix
# https://cran.r-project.org/web/packages/cvms/vignettes/Creating_a_confusion_matrix.html
library(cvms)

# Making predictions C5.0 using threshold = 0.5
pred_best_thres_c50 = round(predictions_c50_smote_tuned > .5)


c50_results <- cbind(as.data.frame(ifelse(test_data$is_attributed == 'no',0,1)), 
                     as.data.frame(pred_best_thres_c50)) 
names(c50_results) <- c('observed','predicted')

cm_c50 <- c50_results %>%
  group_by(observed,predicted) %>%
  summarise(n = n())

c50 <- plot_confusion_matrix(cm_c50, 
                      target_col = "observed", 
                      prediction_col = "predicted",
                      counts_col = "n",
                      add_counts = FALSE,
                      font_normalized = font(size = 6),
                      font_col_percentages = font(size = 4),
                      font_row_percentages = font(size = 4))

# Making predictions Gradient Boosting using threshold = 0.5
pred_gbm = round(predictions_gbm > .5)

gbm_results <- cbind(as.data.frame(ifelse(test_data$is_attributed == 'no',0,1)), 
                     as.data.frame(pred_gbm)) 
names(gbm_results) <- c('observed','predicted')

cm_gbm <- gbm_results %>%
  group_by(observed,predicted) %>%
  summarise(n = n())

gbm <- plot_confusion_matrix(cm_gbm, 
                      target_col = "observed", 
                      prediction_col = "predicted",
                      counts_col = "n",
                      add_counts = FALSE,
                      font_normalized = font(size = 6),
                      font_col_percentages = font(size = 4),
                      font_row_percentages = font(size = 4))

grid.arrange(c50, gbm, ncol = 2)


### Predictions in the Best threshold considering ROC

# The best threshold is estimated using Youden's J statistic
# J_youdens = true_positive_rate - false_positive_rate
rocs@y.name
rocs@x.name

## C5.0 model
index1 = which.max(rocs@y.values[[1]] - rocs@x.values[[1]])
threshold_c5model = rocs@alpha.values[[1]][index1]

# making predictions using threshold_c5model
pred_best_thres_c50 = factor(round(predictions_c50_smote_tuned > threshold_c5model), labels = c('no','yes'))

# confusion matrix
confusionMatrix(pred_best_thres_c50, test_data$is_attributed, positive = 'yes', mode = "prec_recall")

## GBM model
index2 = which.max(rocs@y.values[[2]] - rocs@x.values[[2]])
threshold_gbm = rocs@alpha.values[[2]][index2]

# making predictions using threshold_gbm
pred_best_thres_gbm = factor(round(predictions_gbm > threshold_gbm), labels = c('no','yes'))

# confusion matrix
confusionMatrix(pred_best_thres_gbm, test_data$is_attributed, positive = 'yes', mode = "prec_recall")


