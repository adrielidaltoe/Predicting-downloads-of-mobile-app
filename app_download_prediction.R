# Project 1 - Machine learning model to predict the downloads of mobile apps

# Setting the working directory
setwd("D:/FCD/Projetos/Projeto-Feedback1")

# Packages for loading and manipulating the data
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

# Packages for splitting the data and machine learning
library(randomForest)
library(class)
library(e1071)
library(caTools)
library(ROSE)
library(caret)
library(C50)
library(PRROC)

# Importing functions created for this project
source("src/FuncTools.R")

# Importing the train data
train <- fread('train_sample.csv')
str(train)


# The train data contains 100000 rows and 8 columns.
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
# ip_range: ip grouped by bins (0,1.2e+05], (1.2e+05,2.1e+05], (2.1e+05,3.65e+05] -> lables 1, 2, 3 respectively.


# Searching for missing values
sapply(train, function(x){sum(is.na(x))})
# no missing values were detected, but there are rows with no values in column attributed_time
# marked with double quotes, that is why missing values for that column were not returned. At first, 
# this may not be a problem because in further manipulations this will be treated.
# We can also notice that features as ip, app, device, os, channel and is_attibuted are categorical 
# variables. These variables will be further properly transformed into categorical type.

## Adding new features to be used in the descriptive analysis
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

################ Descriptive Analysis ###########################

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
train %>% group_by(is_attributed) %>%
  summarise(n = n()) %>%
  mutate(is_attributed = ifelse(is_attributed == 0, 'Clicks', 'Click and download')) %>%
  ggplot(aes(x = is_attributed, y = n, fill = is_attributed)) +
  geom_bar(stat = 'identity') +
  ggtitle('Number of app downloads') +
  xlab('')+
  ylab('')

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

# Distribution of clicks throughtout the 4 days
train %>% group_by(hour) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = as.factor(hour), y = n)) +
  geom_bar(stat = 'identity', fill = 'salmon1', color = 'salmon4') +
  ggtitle('Distribution of clicks throughtout the four days') +
  xlab('Hour') +
  ylab('Clicks')

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

# Downloads in each minute
train %>% filter(is_attributed == 1) %>%
  group_by(date, minute) %>%
  summarise(n = n()) %>%
  ggplot(aes(x = minute, y = n)) +
  geom_line(color = 'salmon1') +
  geom_point(color = 'salmon4') +
  facet_grid(. ~ date) +
  ggtitle('Downloads per minute') +
  xlab('Minute') +
  ylab('Clicks')

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
  ggplot(aes(x = date, y = total, fill = date)) +
  geom_bar(stat = 'identity', fill = 'tomato2', color = 'tomato4') +
  ggtitle('Number of clicks per day') +
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
  ggtitle('Clicks throughout observed days') +
  xlab('Hour') +
  ylab('') +
  theme(legend.position = "none")

# Now, the distribution of downloads in each day
train %>% filter(is_attributed == 1) %>%
  ggplot(aes(x = hour_download, fill = date_download)) +
  geom_bar() +
  facet_grid(. ~ date_download) + 
  ggtitle('Downloads throughout the observed days') +
  xlab('Hour') +
  ylab('') +
  theme(legend.position = "none")

# Number of downloads per day
train %>% filter(is_attributed == 1) %>%
  group_by(date_download) %>%
  summarise(downloads = n())
# 2017-11-06 presented only 2 downloads which makes sense, once the clicks in this day were between 16:00 to 22:00,
# hour of the day with lower clicks and lower downloads as well.
# Thus, it is noticeable that data is time dependent. 


## Exploring the IP column

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

# Top 10 IPs in clicks
train %>% group_by(ip) %>%
  summarise(n= n()) %>%
  arrange(desc(n)) %>%
  head(10) %>%
  mutate(IP = fct_reorder(as.factor(ip), n)) %>%
  ggplot(aes(x = n, y = IP)) +
  geom_bar(stat = 'identity', color = 'orchid4', fill = 'orchid3') +
  ggtitle('Top 10 IPs in clicks') +
  xlab('Clicks') +
  ylab('')

# Verifying if ips with more clicks presented downloads.

# 6 downloads came from 2 IPs; 221 downloads were made from different IPs.
train %>% filter(is_attributed == '1') %>%
  group_by(ip) %>%
  summarise(downloads_per_ip = n()) %>%
  group_by(downloads_per_ip) %>%
  summarise(number_of_occurrence = n())

# preparing ips with more than 1 click
df <- train %>% group_by(ip) %>%
  summarise(n= n()) %>%
  filter(n > 1)

# inner_join train data with df by ip column
joined_ip <- train %>% filter(is_attributed == '1') %>%
  inner_join(df, by = 'ip')

# 77 downloads were made from IPs with more than 1 click
nrow(joined_ip)

# And 7 downloads were made from IPs with more than 100 clicks, specifically from ips 5314, 5348, which
# presented more than 600 clicks. 6 downloads were made from each of those IPs.
joined_ip %>%
  filter(n > 100) %>%
  group_by(ip, date_download) %>%
  summarize(total = n())

# Thus, IP with higher amount of clicks does not necessarily have more downloads.
# Next visualization shows the distribuition of clicks and downloads throughout the IP numbers
p0 <- ggplot(train, aes(x = seq_along(ip), y = ip)) +
  geom_point(aes(col=factor(is_attributed)), alpha=0.8, size=0.05) +
  labs(x = 'index', y = 'IP')
p0

# It's observed that there are 3 ranges of ips with different concentrations of clicks: 0 - 120000; 120000 - 210000,
# 210000 - 360000 (approximated values). Clicks are more concentrated at the first range and decreases
# gradually in the second and third ranges. On the other hand, the number of downloads are
# dispersed through all IPs, which is shown in the next visualization.
p1 <- train %>% filter(is_attributed == 1) %>%
  ggplot(aes(x = seq_along(ip), y = ip)) +
  geom_point(color = 'steelblue3') +
  labs(x = 'index', y = 'IP') +
  ggtitle('Distribution of downloads in IPs')
p1

# Investigating the number of downloads in each ip range
p2 <- train %>% filter(is_attributed == 1) %>%
  group_by(ip_range) %>%
  count() %>%
  ggplot(aes(x = ip_range, y = n)) + 
  geom_bar(stat = 'identity', fill = 'steelblue4') +
  labs(x = '', y = 'Downloads') +
  ggtitle('Downloads per IP range')
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
# Let's compare the number of clicks and number of downloads for each ip-range.
p3 <- train %>% 
  group_by(ip_range) %>%
  count() %>%
  ggplot(aes(x = ip_range, y = n)) + 
  geom_bar(stat = 'identity', fill = 'steelblue4') +
  labs(x = '', y = 'Clicks') +
  ggtitle('Clicks per IP range')
grid.arrange(p3, p2, ncol = 2)

## Exploring app column

# 37 app id had downloads. The next visualization shows the app ids that presented more than 3 downloads.
train %>% filter(is_attributed == 1) %>%
  group_by(app) %>%
  summarise(total = n()) %>%
  filter(total > 3) %>%
  mutate(App_id = fct_reorder(as.factor(app), total)) %>%
  ggplot(aes(x = total, y = App_id)) +
  geom_bar(stat = 'identity', color ='steelblue4' , fill = 'steelblue3') +
  ggtitle('App id with more than 3 downloads') +
  ylab('App') +
  xlab('Downloads')

# Downloads were made by clicking on app id with more clicks?
app_group <- train %>% group_by(app) %>% summarise(total_clicks = n())
app_group %>% arrange(desc(total_clicks)) # app id 3 had the most clicks, 18279 

# However, the following graph shows that the app id with most downloads was not the ones 
# with the most clicks.
train %>% filter(is_attributed == 1) %>%
  group_by(app) %>%
  summarise(total_downloads = n()) %>%
  inner_join(app_group, by = 'app') %>%
  mutate(app = fct_reorder(as.factor(app), desc(total_downloads))) %>%
  filter(total_downloads > 3) %>%
  ggplot(aes(x = app, y = total_downloads)) +
  geom_bar(stat = 'identity') +
  geom_line(aes(x = app, y = total_clicks/200, group = 1), size = 1, color="blue") +
  geom_point(aes(x = app, y = total_clicks/200), color = 'blue', size = 1.5) +
  scale_y_continuous(sec.axis = sec_axis(~.*200, name = "Total clicks")) +
  theme(axis.text.y.right = element_text(color = "blue")) +
  ggtitle('App id with more than 3 downloads') +
  ylab('Downloads') +
  xlab('App id')

## Exploring device column

# 25 of 100 devices had downloads. Device 0 and 1 had the most downloads.
train %>% filter(is_attributed == 1) %>%
  group_by(device) %>%
  summarise(total = n()) %>%
  mutate(factor = fct_reorder(as.factor(device), total)) %>%
  head(10) %>%
  ggplot(aes(x = total, y = factor)) +
  geom_bar(stat = 'identity', color = 'turquoise4', fill = 'turquoise3') +
  ggtitle('Device id with downloads') +
  ylab('Device') +
  xlab('Downloads')

# Next visualization presents a comparison between the devices that presented the most clicks and 
# the number of downloads with those devices. From the barplot we notice that around 94% of the clicks
# came from device 1 and 4.3% from device 2. Device 1 was also responsible for the most downloads (64%)
# followed by device 0 with 23%.
train %>% group_by(is_attributed) %>%
  count(device) %>%
  arrange(desc(n), .by_group = TRUE) %>%
  filter(n > 2) %>%
  ggplot(aes(x = as.factor(device), y = n, fill = as.factor(device))) +
  geom_bar(stat = 'identity') +
  geom_text(aes(label=n), position=position_dodge(width=0.9), vjust=-0.25) +
  theme(legend.position = "none") +
  facet_grid(.~ is_attributed) +
  ggtitle('Clicks per device vs. downloads per device') +
  ylab('')+
  xlab('Device')


## Exploring os column
 
# Os columns has 130 unique values. OS 19 had the most clicks and also the most downloads.
p3 <- train %>% group_by(os) %>%
  count() %>%
  arrange(desc(n)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(as.factor(os), -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'steelblue3') +
  xlab('os') +
  ggtitle('5 os with most clicks')

p4 <- train %>% filter(is_attributed == 1) %>%
  group_by(os) %>%
  count() %>%
  arrange(desc(n)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(as.factor(os), -n), y = n)) +
  geom_bar(stat = 'identity', fill = 'turquoise4') +
  xlab('os') +
  ggtitle('5 os with most downloads')
grid.arrange(p3, p4, ncol = 2)


## Exploring channel column

# Channel has 161 unique values

# Let's look at the distribution of clicks in those channels. All 161 channels had clicks, next visualization
# shows top 10 channels in clicks.
train %>% group_by(channel) %>%
  count(channel) %>%
  mutate(channel = as.factor(channel)) %>%
  arrange(desc(n)) %>%
  head(10) %>%
  ggplot(aes(x = reorder(channel, -n), y = n)) +
  geom_bar(stat = 'identity', color = 'wheat4', fill = 'thistle4') +
  ggtitle('Top Channels') +
  xlab('Channel') +
  ylab('')

# Let's look at the distribution of click and downloads within those channels.

df <- train %>% group_by(channel) %>%
  count(channel)

train %>% filter(is_attributed == 1) %>%
  group_by(channel) %>%
  summarise(total = n()) %>%
  inner_join(df, 'channel') %>%
  filter(total > 2) %>%
  ggplot(aes(x = reorder(as.factor(channel), -total), y = total)) +
  geom_bar(stat = 'identity') +
  geom_line(aes(x = as.factor(channel), y = n/10, group = 1), size = 1, color="blue") +
  geom_point(aes(x = as.factor(channel), y = n/10), color = 'blue', size = 1.5) +
  scale_y_continuous(sec.axis = sec_axis(~.*10, name = "Total clicks")) +
  theme(axis.text.y.right = element_text(color = "blue")) +
  ggtitle('Channels with more than 2 downloads') +
  ylab('Downloads') +
  xlab('Channel')
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
         hour = as.factor(hour),
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

train_cat <- train_cat %>%
  mutate(ip_wday_h = as.factor(ip_wday_h),
         ip_h_channel = as.factor(ip_h_channel),
         ip_h_os = as.factor(ip_h_os),
         ip_hour_app = as.factor(ip_hour_app),
         ip_h_dev = as.factor(ip_h_dev))

# The next functions group the categories of app, channel, os, device variables. group_categories() 
# mantain all the categories that have more than 10 downloads and group the rest. he new variables 
# created by this function are app_group, channel_group, os_group and device_group. group_target_prop(),
# instead of using the download absolut frequency, groups the categories according to the relative 
# frequency of downloads. It uses the minimum and maximum relative frequencies to divide the categories 
# into 5 groups. The variables created by this function are app_resp_group, channel_resp_group, 
# os_resp_group, device_resp_group.
names <- c('app','channel','os','device')
train_cat <- group_categories(train_cat, names, 'is_attributed')
train_cat <- group_target_prop(train_cat, names, 'is_attributed')
View(train_cat)

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

# The p-value from x2 test shows that is_attributed, date, hour, wday and minute are independent of each other, 
# which is confirmed by the association coefficients of the tests applied.
# The correlation plot illustrates the asymmetry: is_attributed is somehow predicted if
# the value of some x variables is known, but much less information can be retrived by only knowing
# is_attributed. The association x -> y is stronger with variables app and channel, and weaker
# with device and os. The variable date presented no association with is_attributed.Furthermore, 
# channel and app have strong association; and device presented moderate association with add, channel
# and os. The lack of symmetry in this data is clearly shown in the theil_sym visualization, where
# only app and channel presented symmetry, i.e. app can be somewhat determined if channel is known, and
# vice-versa. 
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


##################### Data split ##################

train_model <- train_cat %>% select(-c(date))

set.seed(1045)
sample <- sample.split(train_model$is_attributed, SplitRatio = 0.70)

train_set <- subset(train_model, sample == TRUE)
test <- subset(train_model, sample == FALSE)

dim(train_set)
dim(test)
table(test$is_attributed)

#################### Feature Selection ###############

# Feature selection is important for two reasons: first, we have to keep the model as simple as
# possible; and second, using insignificant variables can impact the model performance.

# Methods: 
# 1. filter methods: correlation/association, hypothesis test, information gain
# 2. wrapper methods: forward selection (variables are being added one by one), backward selection 
# (variables are droped one by one), stepwise selection (both forward and backward, i.e., at each
# iteration a variable can be added or droped from the model)
# 3. embedded method

# Evaluating the most important features using Random Forest, the original variables are not
# used because they have more than 53 categorical variables.
# Using random forest with importance = TRUE

names1 <- names(train_set)[-c(1:6,22:25)]
names2 <- names(train_set)[-c(1:6,18:21)]
var = list(names1, names2)

for(names in var){
  set.seed(1045)
  feature_imp <- randomForest(x=train_set[, ..names], y=train_set$is_attributed,
                              importance = TRUE)
  varImpPlot(feature_imp)
}

########## Training C5.0 model ##########################

##### Using features selected with Random Forest ###########

### Mean decrease accuracy
var = names(train_set)[c(18:21,9,8,10,12)]

set.seed(1045)
C50_rf_mda <- C5.0(x = train_set[,..var] , y = train_set$is_attributed, trials = 10)

evaluate_model(model = C50_rf_mda, test_set = test, variables = var, target_name = 'is_attributed')
# ROC-AUC: 0.81, PR-AUC: 0.16, precision: 0.25, recall: 0.63, F1: 0.36.

## Balancing data using ROSE
balanced <- ROSE(is_attributed ~.,data = train_set[,c(6,18:21,9,8,10,12)])$data

set.seed(1045)
C50_rf_mda_balanced <- C5.0(x = balanced[,var] , y = balanced$is_attributed, trials = 10)

evaluate_model(model = C50_rf_mda_balanced, test_set = test, variables = var, target_name = 'is_attributed')

# It is observed that balancing the data with Rose did not improve the model.

### Gini 
var = names(train_set)[c(8,18,19,20,11,10)]

set.seed(1045)
C50_rf_gini <- C5.0(x = train_set[,..var] , y = train_set$is_attributed, trials = 10)

evaluate_model(model = C50_rf_gini, test_set = test, variables = var, target_name = 'is_attributed')
# ROC-AUC: 0.81, PR-AUC: 0.16, precision: 0.25, recall: 0.63, F1: 0.36.
# Comparable to Mean Decrease Accuracy model.

## Balanced data 
balanced <- ROSE(is_attributed ~.,data = train_set[,c(6,12,8,18,19,9,20,11,10)])$data

set.seed(1045)
C50_rf_gini_balanced <- C5.0(x = balanced[,-c(1)] , y = balanced$is_attributed, trials = 10)

evaluate_model(model = C50_rf_gini_balanced, test_set = test, variables = names(balanced), target_name = 'is_attributed')



## Variables with sufixe resp_group will not be tested because they presented lower importance than those
# with sufixe group.

################# xgbTree ######################
# To apply this algorithm, the predictor factor variables must be encoded (numeric).
# after that, predictor variables must be converted into Dmatrix, data structure 
# that XGBoost supports and gives it acclaimed performance and efficiency gains.

# XGBoost hyperparameters
# learning_rate: step size shrinkage used to prevent overfitting [0,1], lower bet means robustness
# max_depth: determines how deeply each tree is allowed to grow
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.
# objective: determines the loss function to be used like reg:linear for regression problems,
# reg:logistic for classification problems with only decision, binary:logistic for classification
# problems with probability.
# nrounds: number of passes on the data. The more complex the relationship between the features and the
# label is, the more passes will be needed. Each pass will enhance the model by reducing the difference
# between ground truth and prediction.

# The smaller the score in each node is, the better the structure of the tree is.
# https://xgboost.readthedocs.io/en/latest/R-package/xgboostPresentation.html

# Preparing the data. XGBoost accepts only numeric data. If the data is numeric, then it is only necessary to 
# create a DMatrix and train the xgboost. In cases where variables are categorical, we can create a sparce
# matrix, where all categorical variables will be transformed into numeric features with binary values of
# 0 and 1. This method is called one-hot encoding.

# Evaluating xgboost performance with all variables
var = names(train_set)[c(18:21,9,8,10,12,6)]
balanced <- ROSE(is_attributed ~.,data = train_set[,c(6,18:21,9,8,10,12)])$data

prediction1 <- train_xgboost(target = 'is_attributed', variables = var,
                            train_dataset = balanced, test_dataset = test)

evaluate_model(NULL, test, 'is_attributed', predictions = prediction1$predictions)

# Improvement in recall (0.72) was accomplished with xgboost and ROC-AUC also improved to 0.85.

## Let's try SMOTE to balance the data

# perc_over controls the over-sampling of the minority class
# perc_under controls the under-sampling of the majority class
over_sample = c(100,300,700,1000,1500,2000,2500)

names = names(train_set)[c(18:21,9,8,10,12,6)]
learner = c(1:3)
for(i in learner){
  grid_search_smote(train_data = train_set, test_data = test, target = 'is_attributed', 
                    features = names, oversamples = over_sample, model = i)
}

### Final model #####

# The best model was C5.0 with smote technique to balance the classes.
var = names(train_set)[c(18:21,9,8,10,12,6)]
set.seed(1045)
smoted_data <- SMOTE(is_attributed ~., data = train_set[,..var], perc.over = 1500, k = 5,
                     perc.under = 400)

var1 <- var[var != 'is_attributed']
set.seed(1045)
model_final <- C5.0(x = smoted_data[,..var1] , y = smoted_data$is_attributed, trials = 10)
evaluate_model(model = model_final, test_set = test, variables = var1, target_name = 'is_attributed')
