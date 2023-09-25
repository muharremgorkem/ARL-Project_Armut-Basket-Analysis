##############################################################
# Service Recommendation System Using Association Rule Learning:
# Exploring User Needs on Armut
##############################################################

##############################################################
# 1. Business Problem
##############################################################
# Armut is Turkey's largest online service platform that connects service providers
# with customers seeking various services. Users can conveniently access services
# such as cleaning, renovation, and moving through their computers or smartphones with just a few taps.

# The main objective is to create a product recommendation system using Association Rule Learning
# based on a dataset containing information about users who have utilized services and the services
# and categories they have chosen.

# Dataset story
# The dataset consists of the services that customers have received along with the categories of
# these services. Additionally, it includes the date and time information for each service received.

# Variables:
# UserId: Customer number, representing unique identifiers for each customer.
# ServiceId: Anonymized service identifiers that belong to different categories.
#       For example, a service like "Sofa Cleaning" can fall under the "Cleaning" category.
#       Each ServiceId can be found in different categories and may represent different services
#       under different categories. For instance, a ServiceId with CategoryId 7 might represent
#       "Radiator Cleaning," while the same ServiceId with CategoryId 2 might represent "Furniture Assembly."
# CategoryId: Anonymized category identifiers, representing different service categories
#       such as Cleaning, Moving, Renovation, etc.
# CreateDate: The date when the service was purchased, providing the timestamp for each transaction.

###############################################################
# 2. Data Preparation
###############################################################

# Importing libraries
##############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('Datasets/armut_data.csv')


# Data understanding
##############################################
def check_df(dataframe, head=5):
    print('################# Columns ################# ')
    print(dataframe.columns)
    print('################# Types  ################# ')
    print(dataframe.dtypes)
    print('##################  Head ################# ')
    print(dataframe.head(head))
    print('#################  Shape ################# ')
    print(dataframe.shape)
    print('#################  NA ################# ')
    print(dataframe.isnull().sum())
    print('#################  Quantiles ################# ')
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99]).T)

check_df(df)

# Creating a new variable that represents services by combining
# ServiceID and CategoryID with an underscore "_"
################################################################
df['SC_ID'] = df['ServiceId'].astype(str) + "_" + df['CategoryId'].astype(str)

# The dataset consists of the date and time when services are received, but there is no defined cart (invoice, etc.).
# In order to apply Association Rule Learning, we need to create a cart (invoice, etc.) definition,
# which is the monthly services received by each customer. For instance, customer with ID 7256
# received services 9_4 and 46_4 in August 2017, which represents one cart; and received services 9_4 and 38_4
# in October 2017, which represents another cart. Each cart should be uniquely identified with an ID.
# To achieve this, first, create a new date variable that includes only the year and month information.
# Then, combine UserID and the newly created date variable with an underscore (_) to create a new variable named ID.
#####################################################################################################################
df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df['New_Date'] = pd.to_datetime(df['CreateDate']).dt.to_period('M')
df['UserId'] = df['UserId'].astype(str)
df['BasketId'] = df['UserId'] + '_' + df['New_Date']
df.head()

##########################################
# 3. Association Rules
##########################################
# Creating a pivot table for the 'BasketId' - 'SC_ID'
#######################################################
df_frequent = df.pivot_table("UserId", "BasketId", "SC_ID", aggfunc="count").fillna(0).applymap(lambda x: 1 if x > 0 else 0)

# Creating association rules
############################
def create_rules(dataframe):
    dataframe = create_basket_service_df(dataframe)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

rules = create_rules(df)
rules.sort_values(by="lift",ascending=False)


# Creating the 'arl_recommender' function and make service recommendations for a user
# who has recently received the "2_0" service
#####################################################################################
def arl_recommender(rules_df, hizmet_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, hizmet in enumerate(sorted_rules["antecedents"]):
        for j in list(hizmet):
            if j == hizmet_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    # recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[0:rec_count]

arl_recommender(rules, '2_0', 5)
# ['22_0', '25_0', '15_1', '13_11', '38_4']

#BONUS 1: Rush Hours - Most Purchased Time Zone in Graph
import matplotlib.pyplot as plt
import seaborn as sns
df["Hour"] = df["CreateDate"].dt.hour
hours = [hour for hour, df in df.groupby("Hour")]
plt.plot(hours, df.groupby(["Hour"]).count())
plt.xticks(hours)
plt.grid()
plt.xlabel("hours")
plt.ylabel("number of orders")
plt.show(block=True)

#BONUS 2: Most Purchased Services in Graph
Services = [col for col, df in df.groupby(["SC_ID"])]
plt.plot(Services, df.groupby(["SC_ID"]).count())
plt.xticks(Services, rotation="vertical", size=8)
plt.grid()
plt.xlabel("Services")
plt.ylabel("Number of Orders")
plt.show(block=True)


