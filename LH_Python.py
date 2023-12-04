import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_url = "https://gist.githubusercontent.com/mbonsma/8da0990b71ba9a09f7de395574e54df1/raw/aec88b30af87fad8d45da7e774223f91dad09e88/lh_data.csv"
LH_data = pd.read_csv(data_url)

#check for null values and duplicates
LH_data.info()
LH_data.isna().sum()
LH_data.duplicated().sum()
#no null values or duplicates found, hence proceed.

plt.plot(LH_data.Age, LH_data.Female, label = 'Female', marker = 'o')
plt.plot(LH_data.Age, LH_data.Male,label='Male', marker = 'x')
plt.title('Left-handed vs Age')
plt.xlabel('Age')
plt.ylabel('Left-handed people percentage')
plt.legend()
plt.show()

LH_data['Birth_year'] = 1986 - LH_data.Age 

for i in LH_data:
    LH_data['Mean_lh'] = (LH_data.Male + LH_data.Female)/2
    
#LH_data['Mean_lh'] = LH_data[['Male', 'Female']].mean() is giving a Nan value.. check it!!

plt.plot(LH_data.Birth_year, LH_data.Mean_lh)
plt.title('Mean vs Birth Year')
plt.xlabel('Birth Year')
plt.ylabel('Mean')
plt.show()

1986-min(LH_data.Birth_year) #is the oldest age in the dataset
1986-max(LH_data.Birth_year) #is the youngest age in the dataset
#to calculate Probability of dying at age A given that they are left handed.
#Baye's theorem: P(A|LH) = P(LH|A) * P(A)/P(LH)


def P_lh_given_A(A, study_year = 1990): #P(LH|A)
    early_1900s_rate = LH_data.iloc[67:77, 4].mean()
    late_1900s_rate = LH_data.iloc[0:11, 4].mean()
    middle_rates = LH_data.loc[LH_data['Birth_year'].isin(study_year - A)]['Mean_lh']
#above is the Mean_lh value such that the birth_year is same as study_year - ages_of death.
#isin() used to make the process more efficient as == would take more time and it is less efficient.
    youngest_age = study_year - 1986 + 10 # youngest age when the death study was done. Hence study_year-1986 will be the added age of the youngest person
    oldest_age = study_year - 1986 + 86 # the oldest age is 86
    
    P_return = np.zeros(A.shape) # create an empty array to store the results
#np.zeros() creates an array of zeroes
    # extract rate of left-handedness for people of ages 'ages_of_death'
    P_return[A > oldest_age] = early_1900s_rate / 100 #it is divided by 100 because the data-point is in percentage form
    P_return[A < youngest_age] = late_1900s_rate / 100
    P_return[np.logical_and((A <= oldest_age), (A >= youngest_age))] = middle_rates / 100
    
    return P_return
    
data_url_2 = "https://gist.githubusercontent.com/mbonsma/2f4076aab6820ca1807f4e29f75f18ec/raw/62f3ec07514c7e31f5979beeca86f19991540796/cdc_vs00199_table310.tsv" 
DD_data = pd.read_csv(data_url_2, sep='\t', skiprows=[1])
LH_data.describe()
DD_data.describe()

DD_data.info()
DD_data.duplicated().sum()
DD_data.isna().sum()
DD_data = DD_data.dropna(subset = ['Both Sexes'])

plt.plot(DD_data.Age, DD_data['Both Sexes'], marker = 'o')
plt.title('Age vs Number of people died')
plt.xlabel('Age')
plt.ylabel('No. of People died')
plt.show()  
 
def P_lh(DD_data, study_year = 1990): #here we are trying to find out the probability that the deceased was left-handed. 
#To do so we use summation over age of P(LH/A)* Number of people who died at the age A divided by total number of people died.
     p_list = DD_data['Both Sexes']*P_lh_given_A(DD_data.Age, study_year)
     p = np.sum(p_list)
     p_lh = p / np.sum(DD_data['Both Sexes'])
     return p_lh
print(P_lh(DD_data))

def P_A_given_lh(A, DD_data, study_year=1990):
    P_A = DD_data['Both Sexes'][A]/np.sum(DD_data['Both Sexes'])
    P_lefthanded = P_lh(DD_data, study_year)
    P_lh_A = P_lh_given_A(A, study_year)
    p_final = P_lh_A*P_A/P_lefthanded
    return p_final


def P_A_given_rh(ages_of_death, death_distribution_data, study_year = 1990):
     P_A = death_distribution_data['Both Sexes'][ages_of_death] / np.sum(death_distribution_data['Both Sexes'])
     P_right = 1 - P_lh(death_distribution_data, study_year) # either you're left-handed or right-handed, so P_right = 1 - P_left
     P_rh_A = 1 - P_lh_given_A(ages_of_death, study_year) # P_rh_A = 1 - P_lh_A 
     return P_rh_A*P_A/P_right


A = DD_data.Age
prob_left = P_A_given_lh(A, DD_data, study_year = 1990)
prob_right = P_A_given_rh(A, DD_data, study_year = 1990)
    
plt.plot(A, prob_left, label = 'Left-handedness')
plt.plot(A, prob_right, label = 'Right-handedness')
plt.title('Age of death vs Probability of being at Age')
plt.xlabel('Age of death')
plt.ylabel('Probability of being at Age')
plt.legend()
plt.show()
    
#Task 9: Finding the mean age of death for left and right-handers
mult_left = A*prob_left
average_lh_age = np.nansum(mult_left)

mult_right = A*prob_right
average_rh_age = np.nansum(mult_right)
diffr_age = average_rh_age - average_lh_age

print("Difference in average ages is " +str(diffr_age))
    

