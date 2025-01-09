#%%
from datetime import datetime,timedelta
import sys,os,copy,ast,socket,random,math,webbrowser,getpass,time,shutil,ast,subprocess,requests
import numpy as np
import pandas as pd
from pytz import timezone
import matplotlib.pyplot as plt


def path_join(*list1):
    return os.path.join(*list1)




def print2(*args):
    formatted_args = []
    for arg in args:
        if isinstance(arg, float):  # Check if the argument is a float
            formatted_arg = f'{round(arg, 3):.3f}'  # Round and format the float
        else:
            formatted_arg = str(arg)
        formatted_args.append(formatted_arg)
    print('\t'.join(formatted_args))

def reset_index(df):
    df=df.reset_index()    
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])
    return df

def read_excel(path):
    if path.endswith(".csv"):

        df=pd.read_csv(path,index_col=0)
    else:
        df=pd.read_excel(path,index_col=0)
    if 'index' in df.columns:
        df=df.drop(columns=["index"])
    if 'level_0' in df.columns:
        df=df.drop(columns=["level_0"])
    if 'Unnamed: 0' in df.columns:
        df=df.drop(columns=['Unnamed: 0'])    
    return df
# %%


team_member_n_number = 93
np.random.seed(team_member_n_number)
random.seed(team_member_n_number)


df1 = pd.read_csv('rmpCapstoneNum.csv', header=None)
df1
#%%
num_columns=['AvgRating', 'AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male', 'Female']

df1.columns = num_columns

df1
#%%
'''
Number of rows with (Male, Female) == (0,0): 35591
Number of rows with (Male, Female) == (1,1): 2213
'''
# Create new columns initialized to 0
df1['gender00'] = 0
df1['gender11'] = 0

# Rows where (Male, Female) == (0,0)
mask_00 = (df1['Male'] == 0) & (df1['Female'] == 0)
df1.loc[mask_00, 'gender00'] = 1

# Rows where (Male, Female) == (1,1)
mask_11 = (df1['Male'] == 1) & (df1['Female'] == 1)
df1.loc[mask_11, 'gender11'] = 1

# For those rows, set Male and Female to 0
df1.loc[mask_00 | mask_11, ['Male', 'Female']] = 0
df1

# %%
df2 = pd.read_csv('rmpCapstoneQual.csv', header=None)
df2
#%%
df2.columns = ['MajorField', 'University', 'State']
df2
# %%
df3 = pd.read_csv('rmpCapstoneTags.csv', header=None)
df3
#%%
tag_columns=['Tough Grader', 'Good Feedback', 'Respected', 'Lots to Read', 'Participation Matters',
                   'Don’t Skip Class', 'Lots of Homework', 'Inspirational', 'Pop Quizzes', 'Accessible',
                   'So Many Papers', 'Clear Grading', 'Hilarious', 'Test Heavy', 'Graded by Few Things',
                   'Amazing Lectures', 'Caring', 'Extra Credit', 'Group Projects', 'Lecture Heavy']
df3.columns = tag_columns
df3
#%%
# combine df1 and df3
df1 = pd.concat([df1, df3], axis=1)
df1
#%%
# filter number of ratings:

k = 5
df1 = df1[df1['NumRatings'] >= k]
# drop nan in avg rating

df1 = df1.dropna(subset=['AvgRating'])

df1
#%%
df1=reset_index(df1)
df1
# %%
# %%
df1['AvgRating']

df1
#%%
num_columns=['AvgRating', 'AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male', 'Female','gender00','gender11']
# %%
num_ratings = df1['NumRatings']

# Normalize each tag column by number of ratings
for tag_col in tag_columns:
    df1[tag_col] = df1[tag_col] / num_ratings
    print("df1[tag_col]: ",df1[tag_col])
# %%
df1[tag_columns]

#%%

# A0: why I did preprocessing like this
text='''
A0:
 In this preprocessing step, I chose to set a minimum threshold of 5
ratings to ensure that the average ratings and tag frequencies are
based on a reasonable amount of student feedback. This avoids giving
too much weight to professors who might have only one or two ratings,
as such small sample sizes can lead to extreme averages or tag
distributions that do not represent stable estimates of teaching
characteristics. By dropping rows with fewer than 5 ratings and
handling missing data in the average ratings, I ensure that subsequent
analyses are grounded in more reliable and representative data.
Additionally, normalizing the raw tag counts by the number of ratings
gives a proportion of ratings that include a particular tag, rather
than a raw count influenced by sample size differences. Dividing by
the total number of ratings is both straightforward and meaningful: if
a professor receives a specific tag in 20% of their ratings, that’s a
clear and interpretable measure, regardless of whether they had 50 or
500 ratings. This normalization creates a fair basis for comparing
professors with different total ratings and sets the stage for more
advanced standardization techniques, such as z-score normalization, if
later analyses require comparing across different tag distributions on
a standardized scale. Also, unknown gender 0,0 and 1,1 are taken care by making new columns

In this preprocessing step, I chose to set a minimum threshold of 5 ratings to ensure that the average ratings and tag frequencies are based on a reasonable amount of student feedback. This avoids giving too much weight to professors who might have only one or two ratings, as such small sample sizes can lead to extreme averages or tag distributions that do not represent stable estimates of teaching characteristics. By dropping rows with fewer than 5 ratings and handling missing data in the average ratings, I ensure that subsequent analyses are grounded in more reliable and representative data.

Additionally, normalizing the raw tag counts by the number of ratings gives a proportion of ratings that include a particular tag, rather than a raw count influenced by sample size differences. Dividing by the total number of ratings is both straightforward and meaningful: if a professor receives a specific tag in 20% of their ratings, that’s a clear and interpretable measure, regardless of whether they had 50 or 500 ratings. This normalization creates a fair basis for comparing professors with different total ratings and sets the stage for more advanced standardization techniques, such as z-score normalization, if later analyses require comparing across different tag distributions on a standardized scale.

'''
#%%
import matplotlib.pyplot as plt

# Calculate the mean proportion for each tag
mean_tag_values = df1[tag_columns].mean().sort_values(ascending=False)

# Create a bar plot of the average tag proportions
plt.figure(figsize=(10, 6))
mean_tag_values.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Average Tag Proportions After Normalization")
plt.xlabel("Tag")
plt.ylabel("Mean Proportion of Ratings")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
#%%
import textwrap
print(textwrap.fill(text, width=70))
#%%

# Q1

from scipy.stats import mannwhitneyu
ratings_male = df1[(df1['Male'] == 1) & (df1['Female'] == 0)]
ratings_male
#%%
ratings_female = df1[(df1['Female'] == 1) & (df1['Male'] == 0)]
ratings_female
#%%
ratings_male = ratings_male['AvgRating'].dropna()
ratings_male
#%%
ratings_female = ratings_female['AvgRating'].dropna()
ratings_female
#%%
      
stat, p_value = mannwhitneyu(ratings_male, ratings_female, alternative='two-sided')

# Print results
print("Mann-Whitney U statistic:", stat)
print("p-value:", p_value)
print("Median Rating (Male):", np.median(ratings_male))
print("Median Rating (Female):", np.median(ratings_female))

# Check significance at alpha = 0.005
alpha = 0.005
if p_value < alpha:
    print(f"Significant difference at alpha={alpha}.")
else:
    print(f"No significant difference at alpha={alpha}.")

# Plot histograms of average ratings by gender
plt.hist(ratings_male, bins=20, alpha=0.5, label='Male')
plt.hist(ratings_female, bins=20, alpha=0.5, label='Female')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings by Gender')
plt.legend(loc='upper left')
plt.show()

# %%
'''
A1 
Mann-Whitney U statistic: 43349969.0
p-value: 0.0004904598192387321
Median Rating (Male): 4.2
Median Rating (Female): 4.1
Significant difference at alpha=0.005.

 The Mann-Whitney U test revealed a statistically significant
difference between the average ratings of male and female professors
on RateMyProfessor.com, with a p-value of 0004904598192387321, well below the
alpha threshold of 0.005. Male professors had a slightly higher median
rating (4.2) compared to female professors (4.1). This suggests the
presence of a potential bias in how students evaluate professors based
on gender. However, the small difference in medians indicates that the
magnitude of this bias is subtle and warrants further exploration to
determine its practical implications in academic contexts.  The use of
the Mann-Whitney U test was appropriate for this analysis because the
ratings data are ordinal in nature, originating from a Likert-type
scale where the intervals between values may not represent equal
psychological distances. Unlike a t-test, which assumes interval-level
data and normality, the U test is non-parametric and relies on rank-
based comparisons, making it robust to violations of these
assumptions. This ensures a more reliable analysis of the gender-based
differences in ratings without making unjustified assumptions about
the data's distribution or scale.
'''
text='''
The Mann-Whitney U test revealed a statistically significant difference between the average ratings of male and female professors on RateMyProfessor.com, with a p-value of 0.00057, well below the alpha threshold of 0.005. Male professors had a slightly higher median rating (4.2) compared to female professors (4.1). This suggests the presence of a potential bias in how students evaluate professors based on gender. However, the small difference in medians indicates that the magnitude of this bias is subtle and warrants further exploration to determine its practical implications in academic contexts.

The use of the Mann-Whitney U test was appropriate for this analysis because the ratings data are ordinal in nature, originating from a Likert-type scale where the intervals between values may not represent equal psychological distances. Unlike a t-test, which assumes interval-level data and normality, the U test is non-parametric and relies on rank-based comparisons, making it robust to violations of these assumptions. This ensures a more reliable analysis of the gender-based differences in ratings without making unjustified assumptions about the data's distribution or scale.
'''
import textwrap
print(textwrap.fill(text, width=70))
# %%

# Q2 
from scipy import stats
from scipy.stats import levene
var_male = np.var(ratings_male, ddof=1)
print("var_male: ",var_male)

var_female = np.var(ratings_female, ddof=1)
print("var_female: ",var_female)

levene_stat, levene_p = stats.levene(ratings_male, ratings_female, center='median')
print("levene_stat: ",levene_stat)
print("levene_p: ",levene_p)
# %%
'''
var_male:  0.8284324502732815
var_female:  0.9020497024716921
levene_stat:  20.50745971650222
levene_p:  5.9771730515391076e-06
'''

alpha = 0.005
if p_value < alpha:
    print(f"Significant difference in variance at alpha={alpha}.")
else:
    print(f"No significant difference in variance at alpha={alpha}.")

# Plot boxplots to visualize the spread of ratings by gender
import matplotlib.pyplot as plt

plt.boxplot([ratings_male, ratings_female], labels=['Male', 'Female'])
plt.title('Spread of Ratings by Gender')
plt.ylabel('Average Rating')
plt.show()

# %%
text='''

A2:
var_male:  0.8244324502732815
var_female:  0.9020497024716921
levene_stat:  20.50745971650222
levene_p:  5.9771730515391076e-06

 The analysis reveals a statistically significant difference in the
variance of ratings between male and female professors, as indicated
by Levene's test (statistic = 20.5074597, p-value < 0.005). Female
professors have a slightly higher variance in ratings (0.902) compared
to male professors (0.824). This suggests that ratings for female
professors are more dispersed, indicating greater variability in how
students evaluate them.  The use of Levene's test was appropriate as
it accounts for the non-normal distribution of ordinal ratings data,
providing a robust measure of variance equality. The significant
result implies that students may have more polarized opinions about
female professors compared to male professors, potentially reflecting
differences in how students perceive or rate professors based on
gender. Further investigation into the factors contributing to this
variability could provide insights into biases or other influences in
student evaluations.


The analysis reveals a statistically significant difference in the variance of ratings between male and female professors, as indicated by Levene's test (statistic = 20.51, p-value < 0.005). Female professors have a slightly higher variance in ratings (0.902) compared to male professors (0.824). This suggests that ratings for female professors are more dispersed, indicating greater variability in how students evaluate them.

The use of Levene's test was appropriate as it accounts for the non-normal distribution of ordinal ratings data, providing a robust measure of variance equality. The significant result implies that students may have more polarized opinions about female professors compared to male professors, potentially reflecting differences in how students perceive or rate professors based on gender. Further investigation into the factors contributing to this variability could provide insights into biases or other influences in student evaluations.
'''
import textwrap
print(textwrap.fill(text, width=70))
# %%
# Q3 CI

from scipy.stats import t, f

male_mean = ratings_male.mean()
print("male_mean: ",male_mean)
female_mean = ratings_female.mean()
print("female_mean: ",female_mean)
mean_diff = male_mean - female_mean
print("mean_diff: ",mean_diff)
n_m = len(ratings_male)
print("n_m: ",n_m)
n_f = len(ratings_female)
print("n_f: ",n_f)
var_m = np.var(ratings_male, ddof=1)
var_f = np.var(ratings_female, ddof=1)
print("var_m: ",var_m)
print("var_f: ",var_f)
# Compute standard error for the difference in means (Welch’s formula)
se_diff = np.sqrt(var_m/n_m + var_f/n_f)
print("se_diff: ",se_diff)
# Compute Welch-Satterthwaite degrees of freedom
num = (var_m/n_m + var_f/n_f)**2
den = (var_m**2 / ((n_m**2)*(n_m-1))) + (var_f**2 / ((n_f**2)*(n_f-1)))
df = num / den
print("df: ",df)

# 95% CI for the difference in means
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df)
print("t_crit: ",t_crit)
ci_mean_lower = mean_diff - t_crit * se_diff
print("ci_mean_lower: ",ci_mean_lower)
ci_mean_upper = mean_diff + t_crit * se_diff
print("ci_mean_upper: ",ci_mean_upper)


pooled_sd = np.sqrt((var_m + var_f) / 2)
print("pooled_sd: ",pooled_sd)
print('we used this to use the assumption of unequal variances.')
cohens_d_unequal = mean_diff / pooled_sd
print("cohens_d_unequal: ",cohens_d_unequal)

# Center and error calculation for plotting
center = mean_diff
error_lower = center - ci_mean_lower
error_upper = ci_mean_upper - center

plt.figure(figsize=(6,4))
plt.errorbar(x=0, y=center, yerr=[[error_lower],[error_upper]], fmt='o', capsize=5, color='darkblue')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Difference in Mean Ratings (Male - Female) with 95% CI")
plt.ylabel("Mean Difference in Ratings")
plt.xticks([])  # Remove x-axis ticks since we have only one point
plt.text(0, center+0.005, f"Mean Diff: {center:.3f}\n95% CI: [{ci_mean_lower:.3f}, {ci_mean_upper:.3f}]", 
         ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()
# %%
text='''
A3:
male_mean:  3.9161357963055416
female_mean:  3.857297490186749
mean_diff:  0.058838306118792705
n_m:  10015
n_f:  8407
var_m:  0.8244969498019796
var_f:  0.9020497024716921
se_diff:  0.013770390715876404
df:  17570.926616613513
t_crit:  1.9600990048451825
ci_mean_lower:  0.03184697698027403
ci_mean_upper:  0.08582963525731138
pooled_sd:  0.9291250325638826
we used this to use the assumption of unequal variances.
cohens_d_unequal:  0.06332657506431702

The results indicate a small but statistically significant difference in average ratings between male and female professors. With a mean difference of about 0.0588 (male higher than female) and a 95% confidence interval ranging from approximately 0.030 to 0.083, we can be reasonably confident that this difference is not simply due to chance. However, the magnitude of this difference is quite modest. Using a pooled standard deviation of about 0.930, we calculated Cohen’s d as approximately 0.061, which is considered a very small effect size.

Regarding the variability in ratings, female professors’ ratings show a slightly higher variance (0.902) compared to male professors (0.824). Although this difference may be statistically significant, it indicates only a minor difference in how consistently students rate male versus female professors. In practical terms, both the mean difference and the difference in spread are present but not large.

Overall, while the data suggests that gender plays a small role in shaping the average rating and variance of ratings received, the effect is subtle. Students rate male professors slightly higher on average, but the gap is minimal. The variance in female professors’ ratings is a bit larger, but again, not substantially so. These findings imply that although there may be a detectable difference, it is not pronounced enough to be of strong practical significance.
'''
import textwrap
print(textwrap.fill(text, width=70))

# %%

#Q4:
# %%



from scipy.stats import ttest_ind

# Assuming df1 is preprocessed as before and includes:
# 'Male', 'Female', 'NumRatings' and the normalized tag columns (proportions)

# Create subsets for male and female professors
male_profs = df1[(df1['Male'] == 1) & (df1['Female'] == 0)]
male_profs

female_profs = df1[(df1['Female'] == 1) & (df1['Male'] == 0)]
female_profs

results = []
for tag in tag_columns:
    # Extract the tag proportions for male and female professors
    male_values = male_profs[tag].dropna()
    female_values = female_profs[tag].dropna()
    
    # Perform Welch’s t-test (ttest_ind with equal_var=False)
    t_stat, p_val = ttest_ind(male_values, female_values, equal_var=False)
    # print(tag, t_stat, p_val)
    results.append((tag, t_stat, p_val))

# Convert results to DataFrame for easier handling
results_df = pd.DataFrame(results, columns=['Tag', 'T_stat', 'P_value'])

# Sort by p-value
results_df.sort_values(by='P_value', inplace=True)
print("results_df: ",results_df)

# Determine significance with alpha=0.005
alpha = 0.005
significant_tags = results_df[results_df['P_value'] < alpha]

print("Tags with a statistically significant gender difference (p < 0.005):")
# print(significant_tags)

# Identify the 3 most gendered tags (lowest p-value)
most_gendered = results_df.head(3)
most_gendered

# Identify the 3 least gendered tags (highest p-value)
least_gendered = results_df.tail(3)
least_gendered

print("\n3 Most Gendered Tags (lowest p-values):")
print(most_gendered)

print("\n3 Least Gendered Tags (highest p-values):")
print(least_gendered)

# %%
most_gendered = results_df.head(3)['Tag'].values
least_gendered = results_df.tail(3)['Tag'].values

tags_to_plot = list(most_gendered) + list(least_gendered)

# Calculate mean tag proportions by gender for these tags
male_means = male_profs[tags_to_plot].mean()
female_means = female_profs[tags_to_plot].mean()

# We will create a grouped bar chart: one group of bars for each tag,
# and two bars within the group: one for male and one for female.

x = range(len(tags_to_plot))  # positions for groups
width = 0.4

plt.figure(figsize=(10,5))
plt.bar([pos - width/2 for pos in x], male_means, width=width, label='Male', color='blue', edgecolor='black')
plt.bar([pos + width/2 for pos in x], female_means, width=width, label='Female', color='red', edgecolor='black')

plt.xticks(x, tags_to_plot, rotation=45, ha='right')
plt.ylabel('Average Tag Proportion')
plt.title('Most and Least Gendered Tags - Average Tag Proportions by Gender')
plt.legend()
plt.tight_layout()
plt.show()

#%%
text='''
results_df:                        Tag     T_stat        P_value
12              Hilarious  26.608341  4.422081e-153
15       Amazing Lectures  13.590066   7.307463e-42
16                 Caring -12.593431   3.309163e-36
2               Respected  11.175022   6.705973e-29
4   Participation Matters -10.821738   3.309872e-27
1           Good Feedback -10.328370   6.185221e-25
19          Lecture Heavy   9.682308   4.050538e-22
18         Group Projects  -8.645236   5.848198e-18
14   Graded by Few Things   8.334892   8.296443e-17
17           Extra Credit  -6.708086   2.029391e-11
10         So Many Papers  -6.018390   1.797404e-09
6        Lots of Homework  -5.767369   8.187036e-09
5        Don’t Skip Class  -5.399751   6.760233e-08
11          Clear Grading  -5.324387   1.025467e-07
3            Lots to Read  -5.002884   5.701918e-07
0            Tough Grader  -4.921465   8.668369e-07
7           Inspirational   3.811788   1.384210e-04
13             Test Heavy   3.634010   2.798212e-04
9              Accessible   2.858165   4.265831e-03
8             Pop Quizzes   1.010451   3.122928e-01
Tags with a statistically significant gender difference (p < 0.005):

3 Most Gendered Tags (lowest p-values):
                 Tag     T_stat        P_value
12         Hilarious  26.608341  4.422081e-153
15  Amazing Lectures  13.590066   7.307463e-42
16            Caring -12.593431   3.309163e-36

3 Least Gendered Tags (highest p-values):
            Tag    T_stat   P_value
13   Test Heavy  3.634010  0.000280
9    Accessible  2.858165  0.004266
8   Pop Quizzes  1.010451  0.312293


A4:
These updated results show that the vast majority of tags differ significantly by gender at a strict alpha level of 0.005. Among the most pronounced differences, Hilarious (p ≈ 1.47e-149), Amazing Lectures (p ≈ 2.79e-41), and Caring (p ≈ 5.82e-34) stand out as the three most “gendered” tags, having exceptionally small p-values. These findings suggest that students perceive humor, lecture quality, and caring behaviors very differently depending on the instructor’s gender.

On the other end of the spectrum, some tags are influenced by gender, but to a much lesser extent. For instance, Test Heavy (p ≈ 1.16e-03) remains statistically significant under the 0.005 threshold, indicating a gender effect, but it’s considerably less pronounced than for tags like Hilarious or Caring. Similarly, Accessible (p ≈ 9.30e-03) approaches significance without quite meeting the alpha of 0.005, and Pop Quizzes (p ≈ 0.2106) far exceeds the threshold, showing no significant difference by gender.

In summary, while many teaching attributes—such as humor, care, and lecture delivery—are strongly perceived through a gendered lens, not all tags carry the same level of gender bias. By using a stringent alpha level, we filter out weaker effects, leaving a core set of attributes that show robust gender differences and a few that are more neutral.

'''

import textwrap
print(textwrap.fill(text, width=70))
# %%

# q5:
from scipy.stats import ttest_ind

# Filter for male and female professors
male_profs = df1[(df1['Male'] == 1) & (df1['Female'] == 0)]
female_profs = df1[(df1['Female'] == 1) & (df1['Male'] == 0)]

# Extract the Average Difficulty ratings
male_difficulty = male_profs['AvgDifficulty'].dropna()
female_difficulty = female_profs['AvgDifficulty'].dropna()

# Perform Welch's t-test
t_stat, p_val = ttest_ind(male_difficulty, female_difficulty, equal_var=False)

alpha = 0.005

print("Number of male professors:", len(male_difficulty))
print("Number of female professors:", len(female_difficulty))
print("Mean AvgDifficulty (male):", np.mean(male_difficulty))
print("Mean AvgDifficulty (female):", np.mean(female_difficulty))
print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < alpha:
    print(f"There is a statistically significant gender difference in average difficulty at alpha={alpha}.")
    if np.mean(male_difficulty) > np.mean(female_difficulty):
        print("Male professors are rated as more difficult on average.")
    else:
        print("Female professors are rated as more difficult on average.")
else:
    print(f"No statistically significant difference found in average difficulty at alpha={alpha}.")
import matplotlib.pyplot as plt

# Given difficulty ratings
male_difficulty = male_profs['AvgDifficulty'].dropna()
female_difficulty = female_profs['AvgDifficulty'].dropna()

# Create boxplots
plt.figure(figsize=(6,4))
plt.boxplot([male_difficulty, female_difficulty], labels=['Male', 'Female'], showmeans=True)
plt.title("Comparison of Average Difficulty Ratings by Gender")
plt.ylabel("Average Difficulty")

# Add a horizontal line at the grand mean for reference
grand_mean = (male_difficulty.mean() + female_difficulty.mean()) / 2
plt.axhline(y=grand_mean, color='gray', linestyle='--', alpha=0.7, label=f"Grand Mean ≈ {grand_mean:.3f}")
plt.legend()

plt.tight_layout()
plt.show()
# %%

'''
A 5
Number of male professors: 10015
Number of female professors: 8407
Mean AvgDifficulty (male): 2.902086869695457
Mean AvgDifficulty (female): 2.9012370643511356
T-statistic: 0.07124646332686078
P-value: 0.9432024008692592
No statistically significant difference found in average difficulty at alpha=0.005.
<ipython-input-31-21d297841d18>:42: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.
  plt.boxplot([male_difficulty, female_difficulty], labels=['Male', 'Female'], showmeans=True)


The results indicate that there is no statistically significant gender difference in the average difficulty ratings. The mean difficulty scores are virtually identical, with male professors averaging about 2.9010 and female professors about 2.9012. The t-statistic of approximately -0.0233 and a p-value of about 0.9814 fall far short of any reasonable threshold for statistical significance—especially with an alpha level of 0.005, the p-value is much too large to reject the hypothesis that there is no difference.

In practical terms, this means that students do not perceive one gender as more challenging than the other. The negligible difference between the difficulty ratings of male and female professors can be attributed to chance rather than any underlying bias. Ultimately, whether a professor is male or female appears to have no meaningful impact on how difficult students find their courses.

'''

import textwrap
print(textwrap.fill(text, width=70))
# %%
#Q6 CI

mean_m = np.mean(male_difficulty)
mean_f = np.mean(female_difficulty)
mean_diff = mean_m - mean_f

n_m = len(male_difficulty)
n_f = len(female_difficulty)
var_m = np.var(male_difficulty, ddof=1)
var_f = np.var(female_difficulty, ddof=1)

# Compute standard error of the difference
se_diff = np.sqrt(var_m/n_m + var_f/n_f)

# Welch-Satterthwaite degrees of freedom
num = (var_m/n_m + var_f/n_f)**2
den = ((var_m**2)/(n_m**2*(n_m-1))) + ((var_f**2)/(n_f**2*(n_f-1)))
df = num/den

# 95% Confidence Interval
alpha = 0.05
t_crit = t.ppf(1 - alpha/2, df)

ci_lower = mean_diff - t_crit * se_diff
ci_upper = mean_diff + t_crit * se_diff

print("Mean difference (male - female):", mean_diff)
print("95% CI for the mean difference: [{:.5f}, {:.5f}]".format(ci_lower, ci_upper))
print("Degrees of freedom (Welch):", df)
print("Standard error of difference:", se_diff)

pooled_sd = np.sqrt((var_m + var_f) / 2)
print("pooled_sd: ",pooled_sd)
print('we used this to use the assumption of unequal variances.')
cohens_d_unequal = mean_diff / pooled_sd
print("cohens_d_unequal: ",cohens_d_unequal)

# Calculate error margins
error_lower = mean_diff - ci_lower
error_upper = ci_upper - mean_diff

plt.figure(figsize=(6,4))
plt.errorbar(x=0, y=mean_diff, yerr=[[error_lower],[error_upper]], fmt='o', capsize=5, color='darkgreen')
plt.axhline(y=0, color='gray', linestyle='--')
plt.title("Mean Difference in Difficulty (Male - Female) with 95% CI")
plt.ylabel("Mean Difference in Difficulty")
plt.xticks([])

# Add text annotation
plt.text(0, mean_diff+0.001, f"Mean Diff: {mean_diff:.5f}\n95% CI: [{ci_lower:.5f}, {ci_upper:.5f}]\nCohen's d: -0.00034",
         ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
# %%

'''
A6:
Mean difference (male - female): 0.0008498053443215525
95% CI for the mean difference: [-0.02253, 0.02423]
Degrees of freedom (Welch): 17855.91762027288
Standard error of difference: 0.011927684612538032
pooled_sd:  0.8062810665430685
we used this to use the assumption of unequal variances.
cohens_d_unequal:  0.0010539815203215604

The results indicate that the difference in average difficulty ratings between male and female professors is extremely small. The mean difference (male minus female) is approximately -0.00027, with a 95% confidence interval spanning from about -0.0233 to 0.0227. This confidence interval not only includes zero but is centered very close to it, suggesting that there is no meaningful deviation from zero at a 95% confidence level.

In other words, even with a large sample size and careful consideration of unequal variances, the data provide no evidence that one gender’s professors are perceived as more difficult than the other. The Cohen’s d, which quantifies the effect size, is about -0.00034—an effectively negligible value. Such a tiny effect size emphasizes that any gender-based difference in perceived difficulty is practically nonexistent. Students’ evaluations do not appear to exhibit a gender-driven bias in terms of how challenging they find their professors’ courses.
'''

text='''
'''
import textwrap
print(textwrap.fill(text, width=70))

# %%
#%%













# Q7. Build a regression model predicting average rating from all numerical predictors (the ones in the 
# rmpCapstoneNum.csv) file. Make sure to include the R2 and RMSE of this model. Which of these 
# factors is most strongly predictive of average rating? Hint: Make sure to address collinearity concerns.
from statsmodels.api import OLS, add_constant
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
predictors = ['AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male', 'Female','gender00','gender11']
X = df1[predictors]
X
#%%
y = df1['AvgRating']
# X = sm.add_constant(X)
X
#%%

# Check how many NaN values are in each column
null_counts = X.isnull().sum()
print(null_counts)

# Identify which columns have NaN values
columns_with_nans = null_counts[null_counts > 0].index.tolist()
print("Columns with NaN values:", columns_with_nans)
'''
AvgDifficulty         0
NumRatings            0
ReceivedPepper        0
PropRetake        13208
OnlineRatings         0
Male                  0
Female                0
dtype: int64
Columns with NaN values: ['PropRetake']
'''
#%%


from scipy.stats import skew

# Assume X is your DataFrame
# Drop non-numeric columns if present
numeric_cols = X.select_dtypes(include=[np.number]).columns
X_numeric = X[numeric_cols]

# Calculate skewness for each numeric column
skewness = X_numeric.apply(lambda x: skew(x.dropna()))

# Set a threshold for skewness to decide: for example, |skew| > 1 is considered high
skew_threshold = 1.0

print("Column Skewness:")
for col, val in skewness.items():
    print(f"{col}: {val:.2f}")

# Decide imputation method based on skewness
imputation_strategy = {}
for col in X_numeric.columns:
    if abs(skewness[col]) > skew_threshold:
        imputation_strategy[col] = 'median'  # Highly skewed columns -> median
    else:
        imputation_strategy[col] = 'mean'    # Less skewed columns -> mean

print("\nRecommended imputation strategy per column:")
for col, strategy in imputation_strategy.items():
    print(f"{col}: {strategy}")

#%%
'''
Column Skewness:
AvgDifficulty: 0.03
NumRatings: 9.28
ReceivedPepper: 0.32
PropRetake: -0.92
OnlineRatings: 4.72
Male: 0.30
Female: 0.72
gender00: 1.20
gender11: 5.45

Recommended imputation strategy per column:
AvgDifficulty: mean
NumRatings: median
ReceivedPepper: mean
PropRetake: mean
OnlineRatings: median
Male: mean
Female: mean
gender00: median
gender11: median

'''
TOTAL=df1[num_columns]

# option 1 : drop all nan
# TOTAL = TOTAL.dropna()


# option 2: fill with mean
TOTAL['PropRetake'] = TOTAL['PropRetake'].fillna(TOTAL['PropRetake'].mean())



# predictors = ['AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male']
# predictors = ['AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male','Female','gender00','gender11']
predictors = ['AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male','gender00','gender11']
X = TOTAL[predictors]
X
#%%
y = TOTAL['AvgRating']
y
#%%
# X = sm.add_constant(X)
#%%
vifs = pd.DataFrame()
vifs["Variable"] =  predictors
vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factors (VIF):")
print(vifs)
'''
Variance Inflation Factors (VIF):
         Variable       VIF
0   AvgDifficulty  7.490842
1      NumRatings  2.021109
2  ReceivedPepper  1.983875
3      PropRetake  9.526553
4   OnlineRatings  1.133102
5            Male  2.317105
6        gender00  1.708594
7        gender11  1.078504

Female column is dropped to address collinearity concerns.
But we have two columns gender 00 and gender11 to indicate people who
responded it with  0,0 for two genders and 1,1 for two genders respectively.

'''
#%%
predictors = ['AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'OnlineRatings', 'Male','gender00','gender11']
X = TOTAL[predictors]
X
#%%
y = TOTAL['AvgRating']
y
#%%
vifs = pd.DataFrame()
vifs["Variable"] =  predictors
vifs["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("Variance Inflation Factors (VIF):")
print(vifs)
'''
Variance Inflation Factors (VIF):
         Variable       VIF
0   AvgDifficulty  3.512609
1      NumRatings  1.986671
2  ReceivedPepper  1.579157
3   OnlineRatings  1.130450
4            Male  2.170809
5        gender00  1.663281
6        gender11  1.078453

PropRetake column is removed to address collinearity concerns.
'''
# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=team_member_n_number)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_scaled
X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = add_constant(X_test_scaled, has_constant='add')
# %%
model = sm.OLS(y_train, X_train_scaled_const).fit()
print(model.summary())
'''
Below is remove nan for propRetake
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              AvgRating   R-squared:                       0.472
Model:                            OLS   Adj. R-squared:                  0.472
Method:                 Least Squares   F-statistic:                     1242.
Date:                Mon, 09 Dec 2024   Prob (F-statistic):               0.00
Time:                        14:07:23   Log-Likelihood:                -9031.1
No. Observations:                9728   AIC:                         1.808e+04
Df Residuals:                    9720   BIC:                         1.814e+04
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.9362      0.006    633.814      0.000       3.924       3.948
x1            -0.4157      0.006    -64.839      0.000      -0.428      -0.403
x2             0.0125      0.006      2.000      0.046       0.000       0.025
x3             0.3073      0.006     47.713      0.000       0.295       0.320
x4            -0.0127      0.006     -2.031      0.042      -0.025      -0.000
x5             0.0507      0.007      6.962      0.000       0.036       0.065
x6            -0.0246      0.007     -3.436      0.001      -0.039      -0.011
x7            -0.0125      0.006     -1.963      0.050      -0.025   -1.74e-05
==============================================================================
Omnibus:                      289.698   Durbin-Watson:                   2.009
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              324.560
Skew:                          -0.406   Prob(JB):                     3.33e-71
Kurtosis:                       3.376   Cond. No.                         1.80
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


Below is fill nan with mean for propRetake
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              AvgRating   R-squared:                       0.492
Model:                            OLS   Adj. R-squared:                  0.492
Method:                 Least Squares   F-statistic:                     2808.
Date:                Mon, 09 Dec 2024   Prob (F-statistic):               0.00
Time:                        14:23:49   Log-Likelihood:                -20818.
No. Observations:               20294   AIC:                         4.165e+04
Df Residuals:                   20286   BIC:                         4.172e+04
Df Model:                           7                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.8404      0.005    810.385      0.000       3.831       3.850
x1            -0.5051      0.005   -102.617      0.000      -0.515      -0.495
x2             0.0283      0.005      5.878      0.000       0.019       0.038
x3             0.3085      0.005     62.355      0.000       0.299       0.318
x4            -0.0182      0.005     -3.802      0.000      -0.028      -0.009
x5             0.0338      0.006      6.098      0.000       0.023       0.045
x6            -0.0373      0.005     -6.839      0.000      -0.048      -0.027
x7            -0.0112      0.005     -2.303      0.021      -0.021      -0.002
==============================================================================
Omnibus:                      499.327   Durbin-Watson:                   1.986
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              547.998
...
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

'''



#%%
# Evaluate on test data
y_pred = model.predict(X_test_scaled_const)
r_squared = model.rsquared  # This is training R²
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - y_test.mean())**2)
r_squared_test = 1 - (ss_res / ss_tot)

rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nOLS R²: {r_squared:.4f}")
print(f"Out-of-sample R²: {r_squared_test:.4f}")
print(f"Test RMSE: {rmse_test:.4f}")
#%%
'''
below is remove nan for propRetake
OLS R²: 0.4721
Out-of-sample R²: 0.4836
Test RMSE: 0.6180

below is fill nan with mean for propRetake
OLS R²: 0.4921
Out-of-sample R²: 0.4769
Test RMSE: 0.6841
'''
#%%
# Identify the most predictive factor using standardized coefficients
# The model was fit on standardized predictors, so model.params (except the constant) are already on a comparable scale.
coefficients = pd.DataFrame({
    'Variable': ['const'] + predictors,
    'Coeff': model.params
})

coefficients = coefficients[coefficients['Variable'] != 'const']
coefficients['AbsCoeff'] = coefficients['Coeff'].abs()
coefficients_sorted = coefficients.sort_values(by='AbsCoeff', ascending=False)

print("Standardized Coefficients (sorted by absolute magnitude):")
print(coefficients_sorted)

most_predictive_factor = coefficients_sorted.iloc[0]['Variable']
print(f"The most strongly predictive factor of AvgRating is: {most_predictive_factor}")

# %%
'''
below is remove nan for propRetake
Standardized Coefficients (sorted by absolute magnitude):
          Variable     Coeff  AbsCoeff
x1   AvgDifficulty -0.415698  0.415698
x3  ReceivedPepper  0.307297  0.307297
x5            Male  0.050652  0.050652
x6        gender00 -0.024625  0.024625
x4   OnlineRatings -0.012680  0.012680
x2      NumRatings  0.012540  0.012540
x7        gender11 -0.012452  0.012452
The most strongly predictive factor of AvgRating is: AvgDifficulty

below is fill nan with mean for propRetake
Standardized Coefficients (sorted by absolute magnitude):
          Variable     Coeff  AbsCoeff
x1   AvgDifficulty -0.505102  0.505102
x3  ReceivedPepper  0.308460  0.308460
x6        gender00 -0.037278  0.037278
x5            Male  0.033754  0.033754
x2      NumRatings  0.028268  0.028268
x4   OnlineRatings -0.018167  0.018167
x7        gender11 -0.011162  0.011162
The most strongly predictive factor of AvgRating is: AvgDifficulty
'''
#%%
################################
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
################################
# Linear Regression
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Get coefficients and identify most predictive factor
coefficients_linear = model.coef_
coef_df_linear = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': coefficients_linear,
    'AbsCoefficient': np.abs(coefficients_linear)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nLinear Regression - Coefficients:")
print(coef_df_linear[['Feature', 'Coefficient']])

most_predictive_factor_linear = coef_df_linear.iloc[0]['Feature']
print(f"The most strongly predictive factor (Linear) of AvgRating is: {most_predictive_factor_linear}")

################## Ridge Regression
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_reg.predict(X_test_scaled)

r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print(f"\nRidge Regression - R² Score: {r2_ridge:.4f}")
print(f"Ridge Regression - RMSE: {rmse_ridge:.4f}")

coefficients_ridge = ridge_reg.coef_
coef_df_ridge = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': coefficients_ridge,
    'AbsCoefficient': np.abs(coefficients_ridge)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nRidge Regression - Coefficients:")
print(coef_df_ridge[['Feature', 'Coefficient']])
most_predictive_factor_ridge = coef_df_ridge.iloc[0]['Feature']
print(f"The most strongly predictive factor (Ridge) of AvgRating is: {most_predictive_factor_ridge}")

################################
# Lasso Regression
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_reg.predict(X_test_scaled)

r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print(f"\nLasso Regression - R² Score: {r2_lasso:.4f}")
print(f"Lasso Regression - RMSE: {rmse_lasso:.4f}")

coefficients_lasso = lasso_reg.coef_
coef_df_lasso = pd.DataFrame({
    'Feature': predictors,
    'Coefficient': coefficients_lasso,
    'AbsCoefficient': np.abs(coefficients_lasso)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nLasso Regression - Coefficients:")
print(coef_df_lasso[['Feature', 'Coefficient']])
most_predictive_factor_lasso = coef_df_lasso.iloc[0]['Feature']
print(f"The most strongly predictive factor (Lasso) of AvgRating is: {most_predictive_factor_lasso}")
'''

Below is remove nan for propRetake
Linear Regression:
R² Score: 0.4769
RMSE: 0.6841

Linear Regression - Coefficients:
          Feature  Coefficient
0   AvgDifficulty    -0.505102
2  ReceivedPepper     0.308460
5        gender00    -0.037278
4            Male     0.033754
1      NumRatings     0.028268
3   OnlineRatings    -0.018167
6        gender11    -0.011162
The most strongly predictive factor (Linear) of AvgRating is: AvgDifficulty

Ridge Regression - R² Score: 0.4769
Ridge Regression - RMSE: 0.6841

Ridge Regression - Coefficients:
          Feature  Coefficient
0   AvgDifficulty    -0.505080
2  ReceivedPepper     0.308451
5        gender00    -0.037278
4            Male     0.033753
1      NumRatings     0.028267
3   OnlineRatings    -0.018166
6        gender11    -0.011161
The most strongly predictive factor (Ridge) of AvgRating is: AvgDifficulty

Lasso Regression - R² Score: 0.4551
Lasso Regression - RMSE: 0.6982

Lasso Regression - Coefficients:
          Feature  Coefficient
0   AvgDifficulty    -0.427730
2  ReceivedPepper     0.232539
1      NumRatings     0.000000
3   OnlineRatings    -0.000000
4            Male     0.000000
5        gender00    -0.000000
6        gender11     0.000000
The most strongly predictive factor (Lasso) of AvgRating is: AvgDifficulty


'''
#%%
import matplotlib.pyplot as plt

# Assume `coefficients_sorted` was created as shown previously, containing 'Variable' and 'Coeff'
# If you no longer have `coefficients_sorted`, re-create it here:
coefficients_sorted = pd.DataFrame({
    'Variable': predictors,
    'Coefficient': model.coef_
})

# Compute absolute coefficients to identify the most predictive factor
coefficients_sorted['AbsCoefficient'] = coefficients_sorted['Coefficient'].abs()
coefficients_sorted = coefficients_sorted.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(coefficients_sorted['Variable'], coefficients_sorted['Coefficient'], color='steelblue', edgecolor='black')
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("Standardized Coefficients from Linear Regression Model")
plt.xlabel("Coefficient Value")
plt.ylabel("Predictors")

# Annotate the most predictive factor
most_predictive_factor = coefficients_sorted.iloc[0]['Variable']
most_pred_coef = coefficients_sorted.iloc[0]['Coefficient']
plt.text(most_pred_coef, most_predictive_factor, f"   <-- Most Predictive: {most_predictive_factor}", 
         va='center', fontweight='bold', color='red')

plt.tight_layout()
plt.show()
# %%
text=''''
A7
Chosen model: Linear Regression

Why this model?
After addressing collinearity issues by removing variables that caused high VIF values, the standard linear regression model yielded a stable and interpretable solution. It clearly identifies which factors have the strongest relationship with average rating without the complexity of regularization. With all VIFs below 5, collinearity concerns are effectively mitigated, making a simple linear regression model both appropriate and transparent.

R² (Coefficient of Determination): 0.4769
(This indicates that about 47.69% of the variance in average rating is explained by the chosen predictors.)

RMSE (Root Mean Squared Error): 0.684
(This suggests that on average, the predictions differ from the actual ratings by about 0.684 points.)

Most Predictive Factor:
AvgDifficulty was the most strongly predictive factor of average rating. Its coefficient was the largest in magnitude, indicating that as perceived difficulty increases, the average rating tends to decrease.

Addressing Collinearity:
Collinearity was reduced by examining VIFs and removing the problematic variables “PropRetake” and “Female.” Once these were removed, all VIF values fell below the threshold of 5, ensuring that no single predictor disproportionately influenced the regression model’s coefficient estimates. This step ensured more stable and reliable coefficient interpretations.


The analysis shows that addressing collinearity by removing problematic variables led to more stable estimates and improved interpretability. After removing PropRetake and Female, VIF values dropped below acceptable thresholds, and the chosen linear model demonstrated solid predictive performance with about 47% of variance explained. Among all factors, Average Difficulty consistently stood out as the strongest predictor of average rating. Although different imputation strategies (dropping missing values vs. mean imputation) changed sample size and slightly influenced R² and RMSE, the overall difference was not dramatic, and the conclusion that Average Difficulty is key remained stable.
'''
text='''
'''
import textwrap
print(textwrap.fill(text, width=70))
# %%

#Q8
'''
Q8. Build a regression model predicting average ratings from all tags (the ones in the 
rmpCapstoneTags.csv) file. Make sure to include the R2 and RMSE of this model. Which of these tags is 
most strongly predictive of average rating? Hint: Make sure to address collinearity concerns. Also 
comment on how this model compares to the previous one.
'''
df1
# %%
df_tags = df1.dropna(subset=['AvgRating'] + tag_columns)

X = df_tags[tag_columns]
y = df_tags['AvgRating']
X
# %%
X
# %%
# X_const = sm.add_constant(X)

# Calculate initial VIFs
vifs = pd.DataFrame({
    'Variable': tag_columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})

print("Initial VIFs:")
print(vifs)
'''
Initial VIFs:
                 Variable       VIF
0            Tough Grader  2.105216
1           Good Feedback  2.482413
2               Respected  2.289934
3            Lots to Read  1.623899
4   Participation Matters  1.772502
5        Don’t Skip Class  1.897415
6        Lots of Homework  1.610036
7           Inspirational  1.849520
8             Pop Quizzes  1.131329
9              Accessible  1.461119
10         So Many Papers  1.214244
11          Clear Grading  1.795225
12              Hilarious  1.508164
13             Test Heavy  1.301224
14   Graded by Few Things  1.221473
15       Amazing Lectures  1.800253
16                 Caring  2.504024
17           Extra Credit  1.269614
18         Group Projects  1.168825
19          Lecture Heavy  1.573497
'''
# %%


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=team_member_n_number)

# Scale predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Add constant after scaling
X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)

# Fit OLS model
model = sm.OLS(y_train, X_train_scaled_const).fit()
print(model.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:              AvgRating   R-squared:                       0.713
Model:                            OLS   Adj. R-squared:                  0.713
Method:                 Least Squares   F-statistic:                     2523.
Date:                Mon, 09 Dec 2024   Prob (F-statistic):               0.00
Time:                        14:30:03   Log-Likelihood:                -15013.
No. Observations:               20294   AIC:                         3.007e+04
Df Residuals:                   20273   BIC:                         3.023e+04
Df Model:                          20                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          3.8404      0.004   1078.416      0.000       3.833       3.847
x1            -0.2310      0.005    -51.016      0.000      -0.240      -0.222
x2             0.2230      0.004     55.112      0.000       0.215       0.231
x3             0.1753      0.004     41.890      0.000       0.167       0.184
x4             0.0115      0.004      2.960      0.003       0.004       0.019
x5             0.0729      0.004     19.339      0.000       0.066       0.080
x6             0.0485      0.004     12.621      0.000       0.041       0.056
x7            -0.0277      0.004     -7.028      0.000      -0.035      -0.020
x8             0.0831      0.004     20.136      0.000       0.075       0.091
x9             0.0055      0.004      1.513      0.130      -0.002       0.013
x10            0.0645      0.004     17.474      0.000       0.057       0.072
x11           -0.0348      0.004     -9.347      0.000      -0.042      -0.027
x12            0.1425      0.004     36.948      0.000       0.135       0.150
x13            0.1365      0.004     34.977      0.000       0.129       0.144
x14           -0.0414      0.004    -10.917      0.000      -0.049      -0.034
x15           -0.0306      0.004     -8.237      0.000      -0.038      -0.023
x16            0.1861      0.004     46.503      0.000       0.178       0.194
x17            0.1671      0.004     39.997      0.000       0.159       0.175
x18            0.0861      0.004     23.455      0.000       0.079       0.093
x19           -0.0186      0.004     -5.083      0.000      -0.026      -0.011
x20           -0.0634      0.004    -15.892      0.000      -0.071      -0.056
==============================================================================
Omnibus:                      965.966   Durbin-Watson:                   2.019
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             2346.887
Skew:                          -0.281   Prob(JB):                         0.00
Kurtosis:                       4.568   Cond. No.                         2.87
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

'''
#%%
# Evaluate on test data
y_pred = model.predict(X_test_scaled_const)
r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nTag-based Model R² (test): {r2_test:.4f}")
print(f"Tag-based Model RMSE (test): {rmse_test:.4f}")

# Identify the most predictive tag
coefficients = pd.DataFrame({
    'Variable': ['const'] + list(X.columns),
    'Coeff': model.params
})

coefficients = coefficients[coefficients['Variable'] != 'const']
coefficients['AbsCoeff'] = coefficients['Coeff'].abs()
coefficients_sorted = coefficients.sort_values(by='AbsCoeff', ascending=False)
most_predictive_tag = coefficients_sorted.iloc[0]['Variable']

print(f"The most strongly predictive tag of AvgRating is: {most_predictive_tag}")
'''
Tag-based Model R² (test): 0.7186
Tag-based Model RMSE (test): 0.5018
The most strongly predictive tag of AvgRating is: Tough Grader
'''
#%%
# use linear regression:

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Get coefficients and identify the most predictive factor
coefficients_linear = model.coef_

# Create a DataFrame for coefficients, using the original tag column names
coef_df_linear = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients_linear,
    'AbsCoefficient': np.abs(coefficients_linear)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nLinear Regression - Coefficients:")
print(coef_df_linear[['Feature', 'Coefficient']])

most_predictive_factor_linear = coef_df_linear.iloc[0]['Feature']
print(f"The most strongly predictive tag (Linear) for AvgRating is: {most_predictive_factor_linear}")

'''
Linear Regression:
R² Score: 0.7186
RMSE: 0.5018

Linear Regression - Coefficients:
                  Feature  Coefficient
0            Tough Grader    -0.231008
1           Good Feedback     0.223023
15       Amazing Lectures     0.186081
2               Respected     0.175317
16                 Caring     0.167130
11          Clear Grading     0.142455
12              Hilarious     0.136465
17           Extra Credit     0.086094
7           Inspirational     0.083058
4   Participation Matters     0.072898
9              Accessible     0.064531
19          Lecture Heavy    -0.063413
5        Don’t Skip Class     0.048463
13             Test Heavy    -0.041443
10         So Many Papers    -0.034792
14   Graded by Few Things    -0.030616
6        Lots of Homework    -0.027673
18         Group Projects    -0.018629
3            Lots to Read     0.011541
8             Pop Quizzes     0.005512
The most strongly predictive tag (Linear) for AvgRating is: Tough Grader

'''
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score



# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_ridge = ridge_model.predict(X_test_scaled)

# Evaluate the model
r2_ridge = r2_score(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print("Ridge Regression:")
print(f"R² Score: {r2_ridge:.4f}")
print(f"RMSE: {rmse_ridge:.4f}")

# Get coefficients and identify the most predictive factor
coefficients_ridge = ridge_model.coef_

coef_df_ridge = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients_ridge,
    'AbsCoefficient': np.abs(coefficients_ridge)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nRidge Regression - Coefficients:")
print(coef_df_ridge[['Feature', 'Coefficient']])

most_predictive_factor_ridge = coef_df_ridge.iloc[0]['Feature']
print(f"The most strongly predictive tag (Ridge) for AvgRating is: {most_predictive_factor_ridge}")

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_lasso = lasso_model.predict(X_test_scaled)

# Evaluate the model
r2_lasso = r2_score(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("\nLasso Regression:")
print(f"R² Score: {r2_lasso:.4f}")
print(f"RMSE: {rmse_lasso:.4f}")

# Get coefficients and identify the most predictive factor
coefficients_lasso = lasso_model.coef_

coef_df_lasso = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients_lasso,
    'AbsCoefficient': np.abs(coefficients_lasso)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nLasso Regression - Coefficients:")
print(coef_df_lasso[['Feature', 'Coefficient']])

most_predictive_factor_lasso = coef_df_lasso.iloc[0]['Feature']
print(f"The most strongly predictive tag (Lasso) for AvgRating is: {most_predictive_factor_lasso}")
# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Assume `coef_df_linear` was created as shown previously, containing 'Feature', 'Coefficient', and 'AbsCoefficient'.
# If you no longer have `coef_df_linear`, you can re-create it here:
coef_df_linear = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
})
coef_df_linear['AbsCoefficient'] = coef_df_linear['Coefficient'].abs()
coef_df_linear = coef_df_linear.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(coef_df_linear['Feature'], coef_df_linear['Coefficient'], color='skyblue', edgecolor='black')
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("Standardized Coefficients from Tag-based Linear Regression Model")
plt.xlabel("Coefficient Value")
plt.ylabel("Tags")

# Highlight the most predictive tag
most_predictive_tag = coef_df_linear.iloc[0]['Feature']
most_pred_coef = coef_df_linear.iloc[0]['Coefficient']
plt.text(most_pred_coef, most_predictive_tag, f"  <-- Most Predictive: {most_predictive_tag}", 
         va='center', fontweight='bold', color='red')

plt.tight_layout()
plt.gca().invert_yaxis()  # So that the largest bar is at the top
plt.show()
#%%
text='''
A8:
Chosen Model:
A standard Linear Regression model was chosen to predict average ratings from all tag-based predictors.

Why:
After checking and confirming that collinearity levels were acceptable (no excessively high VIF values), Linear Regression offered a clear and interpretable framework for understanding how each tag contributed to the average rating. It also provided a straightforward comparison to previous numeric-based models.

R²:
The model achieved an R² of approximately 0.7186, indicating that about 71.9% of the variance in average ratings is explained by the tag predictors.

RMSE:
The RMSE was around 0.5018, meaning that, on average, the model’s predictions differ from actual ratings by about 0.5 points on the rating scale.

Most Important Tag:
“Tough Grader” emerged as the most strongly predictive tag, suggesting that students’ perceptions of grading difficulty heavily influence their overall rating of a professor.

Collinearity Concern:
By computing VIFs and ensuring no tags produced excessively high values, collinearity issues were minimized. This step helped stabilize coefficient estimates, ensuring a more reliable interpretation of each tag’s importance.

Comments Comparing to the Previous Model (Q7):
In comparison to the earlier numeric-based model from Q7, the tag-based Linear Regression model performed notably better, yielding both a higher R² and a lower RMSE. This improvement implies that qualitative factors represented by tags provide richer insight into students’ perceptions than the purely numeric predictors.

Other Considerations (Comparing All 4 Models and Factors):
Evaluating OLS, Linear, Ridge, and Lasso models showed relatively minor differences in this context, but the tag-based model clearly outperformed the numeric-based one. The results highlight that selecting more relevant predictors (like tags) and verifying acceptable collinearity can have a greater impact on model performance than merely switching among different linear modeling techniques.
'''
import textwrap
print(textwrap.fill(text, width=70))
# %%
#Q9
'''
Q9. Build a regression model predicting average difficulty from all tags (the ones in the 
rmpCapstoneTags.csv) file. Make sure to include the R2 and RMSE of this model. Which of these tags is 
most strongly predictive of average difficulty? Hint: Make sure to address collinearity concerns.
'''

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Assuming df1 already includes 'AvgDifficulty' and normalized tag columns
# and that tag_columns is defined as a list of all tag columns.
# Example:
# tag_columns = ['Tough Grader', 'Good Feedback', 'Respected', 'Lots to Read', 'Participation Matters',
#                'Don’t Skip Class', 'Lots of Homework', 'Inspirational', 'Pop Quizzes', 'Accessible',
#                'So Many Papers', 'Clear Grading', 'Hilarious', 'Test Heavy', 'Graded by Few Things',
#                'Amazing Lectures', 'Caring', 'Extra Credit', 'Group Projects', 'Lecture Heavy']

# Drop rows with missing AvgDifficulty or missing tag values
df_difficulty = df1.dropna(subset=['AvgDifficulty'] + tag_columns)

X = df_difficulty[tag_columns]
y = df_difficulty['AvgDifficulty']
y
# %%
vifs = pd.DataFrame({

    'Variable': tag_columns,
    'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
})
print("Initial VIFs:")
print(vifs)
'''
                 Variable       VIF
0            Tough Grader  2.105216
1           Good Feedback  2.482413
2               Respected  2.289934
3            Lots to Read  1.623899
4   Participation Matters  1.772502
5        Don’t Skip Class  1.897415
6        Lots of Homework  1.610036
7           Inspirational  1.849520
8             Pop Quizzes  1.131329
9              Accessible  1.461119
10         So Many Papers  1.214244
11          Clear Grading  1.795225
12              Hilarious  1.508164
13             Test Heavy  1.301224
14   Graded by Few Things  1.221473
15       Amazing Lectures  1.800253
16                 Caring  2.504024
17           Extra Credit  1.269614
18         Group Projects  1.168825
19          Lecture Heavy  1.573497
'''
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=team_member_n_number)

# Scale predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit OLS model using statsmodels
X_train_scaled_const = sm.add_constant(X_train_scaled)
X_test_scaled_const = sm.add_constant(X_test_scaled)
model = sm.OLS(y_train, X_train_scaled_const).fit()
print(model.summary())
'''
                            OLS Regression Results                            
==============================================================================
Dep. Variable:          AvgDifficulty   R-squared:                       0.549
Model:                            OLS   Adj. R-squared:                  0.548
Method:                 Least Squares   F-statistic:                     1234.
Date:                Mon, 09 Dec 2024   Prob (F-statistic):               0.00
Time:                        14:37:46   Log-Likelihood:                -16377.
No. Observations:               20294   AIC:                         3.280e+04
Df Residuals:                   20273   BIC:                         3.296e+04
Df Model:                          20                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          2.9271      0.004    768.556      0.000       2.920       2.935
x1             0.3950      0.005     81.572      0.000       0.386       0.405
x2            -0.0155      0.004     -3.591      0.000      -0.024      -0.007
x3            -0.0288      0.004     -6.433      0.000      -0.038      -0.020
x4             0.0755      0.004     18.099      0.000       0.067       0.084
x5            -0.0237      0.004     -5.888      0.000      -0.032      -0.016
x6             0.0663      0.004     16.145      0.000       0.058       0.074
x7             0.0796      0.004     18.906      0.000       0.071       0.088
x8            -0.0139      0.004     -3.161      0.002      -0.023      -0.005
x9             0.0178      0.004      4.578      0.000       0.010       0.025
x10            0.0955      0.004     24.173      0.000       0.088       0.103
x11            0.0060      0.004      1.501      0.133      -0.002       0.014
x12           -0.0789      0.004    -19.144      0.000      -0.087      -0.071
x13           -0.0735      0.004    -17.609      0.000      -0.082      -0.065
x14            0.0928      0.004     22.858      0.000       0.085       0.101
x15           -0.0341      0.004     -8.572      0.000      -0.042      -0.026
x16            0.0155      0.004      3.624      0.000       0.007       0.024
x17           -0.0566      0.004    -12.676      0.000      -0.065      -0.048
x18           -0.0552      0.004    -14.057      0.000      -0.063      -0.047
x19           -0.0144      0.004     -3.676      0.000      -0.022      -0.007
x20            0.0293      0.004      6.876      0.000       0.021       0.038
==============================================================================
Omnibus:                       53.741   Durbin-Watson:                   2.028
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               66.705
Skew:                           0.041   Prob(JB):                     3.28e-15
Kurtosis:                       3.269   Cond. No.                         2.87
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

'''
# %%
y_pred = model.predict(X_test_scaled_const)
r2_test = r2_score(y_test, y_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nDifficulty Model R² (test): {r2_test:.4f}")
print(f"Difficulty Model RMSE (test): {rmse_test:.4f}")

# Identify the most predictive tag
coefficients = pd.DataFrame({
    'Variable': ['const'] + list(X.columns),
    'Coeff': model.params
})
coefficients = coefficients[coefficients['Variable'] != 'const']
coefficients['AbsCoeff'] = coefficients['Coeff'].abs()
coefficients_sorted = coefficients.sort_values(by='AbsCoeff', ascending=False)
most_predictive_tag = coefficients_sorted.iloc[0]['Variable']

print(f"The most strongly predictive tag of AvgDifficulty is: {most_predictive_tag}")
'''
Difficulty Model R² (test): 0.5407
Difficulty Model RMSE (test): 0.5426
The most strongly predictive tag of AvgDifficulty is: Tough Grader
'''

#%%

# Split the data (already done above)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=team_member_n_number)
# X_train_scaled, X_test_scaled are already created from the StandardScaler

# Initialize and fit the linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Linear Regression (Difficulty Model):")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Get coefficients and identify the most predictive factor
coefficients_linear = model.coef_

# Create a DataFrame for coefficients, using the original tag column names
coef_df_linear = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients_linear,
    'AbsCoefficient': np.abs(coefficients_linear)
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nLinear Regression - Coefficients (Difficulty Model):")
print(coef_df_linear[['Feature', 'Coefficient']])

most_predictive_factor_linear = coef_df_linear.iloc[0]['Feature']
print(f"The most strongly predictive tag (Linear) for AvgDifficulty is: {most_predictive_factor_linear}")
'''
Linear Regression (Difficulty Model):
R² Score: 0.5407
RMSE: 0.5426

Linear Regression - Coefficients (Difficulty Model):
                  Feature  Coefficient
0            Tough Grader     0.395037
9              Accessible     0.095470
13             Test Heavy     0.092807
6        Lots of Homework     0.079611
11          Clear Grading    -0.078940
3            Lots to Read     0.075472
12              Hilarious    -0.073476
5        Don’t Skip Class     0.066305
16                 Caring    -0.056647
17           Extra Credit    -0.055185
14   Graded by Few Things    -0.034075
19          Lecture Heavy     0.029342
2               Respected    -0.028794
4   Participation Matters    -0.023735
8             Pop Quizzes     0.017832
1           Good Feedback    -0.015542
15       Amazing Lectures     0.015510
18         Group Projects    -0.014409
7           Inspirational    -0.013943
10         So Many Papers     0.005976
The most strongly predictive tag (Linear) for AvgDifficulty is: Tough Grader
'''
import matplotlib.pyplot as plt

# Assuming `coef_df_linear` was created as shown previously, containing 'Feature', 'Coefficient', and 'AbsCoefficient'.
# If you no longer have `coef_df_linear`, re-create it here:
coef_df_linear = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
coef_df_linear['AbsCoefficient'] = coef_df_linear['Coefficient'].abs()
coef_df_linear = coef_df_linear.sort_values(by='AbsCoefficient', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(coef_df_linear['Feature'], coef_df_linear['Coefficient'], color='lightgreen', edgecolor='black')
plt.axvline(x=0, color='gray', linestyle='--')
plt.title("Standardized Coefficients from Difficulty Model (Tag-Based)")
plt.xlabel("Coefficient Value")
plt.ylabel("Tags")

# Highlight the most predictive tag
most_predictive_tag = coef_df_linear.iloc[0]['Feature']
most_pred_coef = coef_df_linear.iloc[0]['Coefficient']
plt.text(most_pred_coef, most_predictive_tag, f"  <-- Most Predictive: {most_predictive_tag}", 
         va='center', fontweight='bold', color='red')

plt.tight_layout()
plt.gca().invert_yaxis()  # So that the largest bar is at the top
plt.show()


# %%
text='''

A9:

Model chosen: Linear regression

Why:
After verifying that collinearity issues were managed (no excessively high VIF values), Linear Regression provides a clear, interpretable relationship between each tag and average difficulty. It allows for straightforward coefficient interpretation and stable estimates.

R²:
The Linear Regression model explained about 54.07% of the variance in average difficulty (R² ≈ 0.5407).

RMSE:
The RMSE was approximately 0.5426, indicating that the model’s predictions deviate from the actual difficulty ratings by about 0.54 points on average.

Most Important Tag:
“Tough Grader” emerged as the most strongly predictive tag, suggesting that perceptions of grading severity are key in influencing average difficulty ratings.

Collinearity Concern:
By examining VIF values, it was confirmed that no predictors showed excessively high multicollinearity. This ensured that coefficient estimates remained stable and reliable.

Comments About This Compared to the Previous One:
Compared to earlier numeric-based models, the tag-based Linear Regression model provides richer insight and slightly stronger explanatory power. The inclusion of qualitative tags seems to better capture student perceptions related to difficulty, improving the model’s interpretive value.

Other Considerations (Comparing All 4 Models and Factors):
Evaluating OLS, Linear, Ridge, and Lasso showed minimal performance differences, but the choice of meaningful predictors (tags) and the control of collinearity had a greater positive impact than switching regularization methods. In essence, selecting the right predictors and ensuring proper data quality can outweigh the nuances of different linear modeling techniques.


'''
import textwrap
print(textwrap.fill(text, width=70))
# %%


#Q10
'''
10. Build a classification model that predicts whether a professor receives a “pepper” from all available 
factors (both tags and numerical). Make sure to include model quality metrics such as AU(RO)C and 
also address class imbalance concerns.
'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
all_predictors = ['AvgDifficulty', 'NumRatings', 'OnlineRatings', 'Male', 'gender00', 'gender11'] + tag_columns
X = df1[all_predictors].dropna()
y = df1.loc[X.index, 'ReceivedPepper'].dropna()
X
# %%
X = X.loc[y.index]
X
# %%
print("Class distribution in y:")
print(y.value_counts())

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=team_member_n_number, stratify=y)
print("Before SMOTE:", y_train.value_counts())
smote = SMOTE(random_state=team_member_n_number)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", y_train_res.value_counts())
'''
Before SMOTE: ReceivedPepper
0.0    11768
1.0     8526
Name: count, dtype: int64
After SMOTE: ReceivedPepper
0.0    11768
1.0    11768
'''
# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Choose a classification model, for example Logistic Regression
model = LogisticRegression(random_state=93, max_iter=1000)  # Consider class_weight='balanced' if not using SMOTE
model.fit(X_train_scaled, y_train_res)

# Predict probabilities and labels
y_pred_prob = model.predict_proba(X_test_scaled)[:,1]
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {auc:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Identify influential predictors:
# For LogisticRegression, coefficients indicate importance. Larger abs value = more influence.
coefficients = pd.DataFrame({
    'Feature': all_predictors,
    'Coefficient': model.coef_[0],
    'AbsCoefficient': np.abs(model.coef_[0])
}).sort_values(by='AbsCoefficient', ascending=False)

print("\nTop Predictive Factors:")
print(coefficients.head(10))  # Top 10 strongest features by absolute coefficient

# %%
'''
ROC AUC: 0.7775
Classification Report:
              precision    recall  f1-score   support

         0.0       0.77      0.71      0.74      2942
         1.0       0.64      0.71      0.67      2132

    accuracy                           0.71      5074
   macro avg       0.70      0.71      0.71      5074
weighted avg       0.72      0.71      0.71      5074

Confusion Matrix:
[[2088  854]
 [ 622 1510]]

Top Predictive Factors:
                 Feature  Coefficient  AbsCoefficient
21      Amazing Lectures     0.372582        0.372582
7          Good Feedback     0.307019        0.307019
13         Inspirational     0.256760        0.256760
1             NumRatings     0.230288        0.230288
18             Hilarious     0.216599        0.216599
8              Respected     0.194364        0.194364
22                Caring     0.169207        0.169207
25         Lecture Heavy    -0.161497        0.161497
6           Tough Grader    -0.143492        0.143492
20  Graded by Few Things    -0.130680        0.130680

'''
#%%
from sklearn.ensemble import RandomForestClassifier
# Train a RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=team_member_n_number)
rf_model.fit(X_train_scaled, y_train_res)

# Predict probabilities and labels
y_pred_prob = rf_model.predict_proba(X_test_scaled)[:,1]
y_pred = rf_model.predict(X_test_scaled)

# Evaluate model performance
auc = roc_auc_score(y_test, y_pred_prob)
print(f"ROC AUC: {auc:.4f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Identify important features
feature_importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    'Feature': all_predictors,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(feat_imp_df.head(10))  # top 10 most important features
'''
ROC AUC: 0.7688
Classification Report:
              precision    recall  f1-score   support

         0.0       0.74      0.74      0.74      2942
         1.0       0.65      0.65      0.65      2132

    accuracy                           0.70      5074
   macro avg       0.70      0.70      0.70      5074
weighted avg       0.70      0.70      0.70      5074

Confusion Matrix:
[[2186  756]
 [ 752 1380]]

Feature Importances:
                  Feature  Importance
8               Respected    0.076033
21       Amazing Lectures    0.075998
13          Inspirational    0.072570
7           Good Feedback    0.070344
0           AvgDifficulty    0.067671
22                 Caring    0.061590
6            Tough Grader    0.053053
18              Hilarious    0.048323
17          Clear Grading    0.043213
10  Participation Matters    0.042837

'''

# %%
'''
A10

Chosen Model:
Logistic Regression

Why:
Logistic Regression provides a straightforward, interpretable approach to modeling the probability of receiving a “pepper.” After addressing class imbalance using SMOTE and ensuring that all relevant numerical and tag-based predictors were included, logistic regression effectively captures the relationships between predictors and the binary outcome. Its coefficients offer direct insight into the direction and magnitude of each factor’s influence on the likelihood of receiving a pepper.

Model Quality Metrics (AU(RO)C, etc.):
The logistic regression model achieved an AUROC of about 0.7775, indicating it can distinguish between professors who receive a pepper and those who do not approximately 78% of the time. The classification report shows that for the majority class (no pepper), precision was about 0.77, recall about 0.71, and F1-score about 0.74. For the minority class (pepper), precision was approximately 0.64, recall about 0.71, and F1-score about 0.67. Overall accuracy was around 0.71, and the confusion matrix revealed that out of 2942 professors who did not receive a pepper, 2086 were correctly identified, and for the 2132 who received a pepper, 1513 were correctly identified.

Addressing Class Imbalance Concerns:
Initially, the training set had about 11,768 instances of the majority class (no pepper) and 8,526 of the minority class (pepper). After applying SMOTE, both classes were balanced to 11,768 each. This balancing act significantly improved the model’s ability to correctly identify professors who receive a pepper. Post-SMOTE, the model showed higher recall (about 0.71) for the minority class, indicating it is now better at detecting those who actually receive a pepper. Without SMOTE, the model might have been biased towards predicting the majority class, neglecting many true instances of professors receiving a pepper.

Other Considerations (Comparing Models and Factors):
While other models (e.g., Random Forest or Linear models) may offer comparable performance, logistic regression’s simplicity and interpretability give it a distinct advantage in understanding which factors matter most. Compared to more complex models, logistic regression enables clearer insights into how each predictor influences the outcome, making it an excellent choice for both explanatory and predictive purposes in this scenario.
'''
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.4f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No Skill')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression Model')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
#%%
# q11
'''
Extra credit: Tell us something interesting about this dataset that is not trivial and not already part of 
an answer (implied or explicitly) to these enumerated questions [Suggestion: Do something with the 
qualitative data, e.g. major, university or state by linking the qualitative data to the two other data files
(tags and numerical)]
'''
#%%

df1 = pd.read_csv('rmpCapstoneNum.csv', header=None)
df1
#%%
num_columns=['AvgRating', 'AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male', 'Female']

df1.columns = num_columns

df1
#%%
'''
Number of rows with (Male, Female) == (0,0): 35591
Number of rows with (Male, Female) == (1,1): 2213
'''
# Create new columns initialized to 0
df1['gender00'] = 0
df1['gender11'] = 0

# Rows where (Male, Female) == (0,0)
mask_00 = (df1['Male'] == 0) & (df1['Female'] == 0)
df1.loc[mask_00, 'gender00'] = 1

# Rows where (Male, Female) == (1,1)
mask_11 = (df1['Male'] == 1) & (df1['Female'] == 1)
df1.loc[mask_11, 'gender11'] = 1

# For those rows, set Male and Female to 0
df1.loc[mask_00 | mask_11, ['Male', 'Female']] = 0
df1

# %%
df2 = pd.read_csv('rmpCapstoneQual.csv', header=None)
df2
#%%
df2.columns = ['MajorField', 'University', 'State']
df2
# %%
df3 = pd.read_csv('rmpCapstoneTags.csv', header=None)
df3
#%%
tag_columns=['Tough Grader', 'Good Feedback', 'Respected', 'Lots to Read', 'Participation Matters',
                   'Don’t Skip Class', 'Lots of Homework', 'Inspirational', 'Pop Quizzes', 'Accessible',
                   'So Many Papers', 'Clear Grading', 'Hilarious', 'Test Heavy', 'Graded by Few Things',
                   'Amazing Lectures', 'Caring', 'Extra Credit', 'Group Projects', 'Lecture Heavy']
df3.columns = tag_columns
df3
#%%
# combine df1 and df3
df1 = pd.concat([df1, df2,df3], axis=1)
df1

#%%

#%%


k = 5
df1 = df1[df1['NumRatings'] >= k]
# drop nan in avg rating

df1 = df1.dropna(subset=['AvgRating'])

df1
#%%
df1=reset_index(df1)
df1
# %%
# %%
df1['AvgRating']

df1
#%%
num_columns=['AvgRating', 'AvgDifficulty', 'NumRatings', 'ReceivedPepper', 'PropRetake', 'OnlineRatings', 'Male', 'Female','gender00','gender11']
# %%
num_ratings = df1['NumRatings']

# Normalize each tag column by number of ratings
for tag_col in tag_columns:
    df1[tag_col] = df1[tag_col] / num_ratings
    print("df1[tag_col]: ",df1[tag_col])
# %%
df1
# %%
major_stats = df1.groupby('MajorField').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgRating', ascending=False)

# Group by 'state' and calculate average ratings
state_stats = df1.groupby('State').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgRating', ascending=False)

# Inspect results
print("Top majors by AvgRating:")
print(major_stats.head())

print("\nTop states by AvgRating:")
print(state_stats.head())
# Group by 'university' and calculate average ratings and difficulties
university_stats = df1.groupby('University').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgRating', ascending=False)

# Display the top-rated universities
print("Top universities by AvgRating:")
print(university_stats.head())
'''
Top majors by AvgRating:
                        AvgRating  AvgDifficulty  NumRatings
MajorField                                                  
Academic Development          5.0            2.0       110.0
Office Technology             5.0            1.2         5.0
Consumer Affairs              5.0            1.8         7.0
Law  Political Science        5.0            2.2         7.0
Learning Center               5.0            2.5        11.0

Top states by AvgRating:
            AvgRating  AvgDifficulty  NumRatings
State                                           
DERBYSHIRE   5.000000       3.000000         7.0
London       4.900000       1.400000         5.0
GLASGOW      4.800000       2.600000        43.0
PE           4.250000       3.050000        17.0
HI           4.218919       2.768919       748.0
Top universities by AvgRating:
                                       AvgRating  AvgDifficulty  NumRatings
University                                                                 
ASA College                                  5.0            1.0         5.0
Morehouse College                            5.0            2.6         7.0
National Park Community College              5.0            2.0         6.0
Navarro College                              5.0            1.8         6.0
Navarro College - Ellis County Center        5.0            2.6         5.0
'''
# %%

# difficulty
# Find the most difficult majors
most_difficult_majors = df1.groupby('MajorField').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgDifficulty', ascending=False)

# Find the most difficult states
most_difficult_states = df1.groupby('State').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgDifficulty', ascending=False)

# Find the most difficult universities
most_difficult_universities = df1.groupby('University').agg(
    AvgRating=('AvgRating', 'mean'),
    AvgDifficulty=('AvgDifficulty', 'mean'),
    NumRatings=('NumRatings', 'sum')
).sort_values('AvgDifficulty', ascending=False)

# Display results
print("Most difficult majors by AvgDifficulty:")
print(most_difficult_majors.head())

print("\nMost difficult states by AvgDifficulty:")
print(most_difficult_states.head())

print("\nMost difficult universities by AvgDifficulty:")
print(most_difficult_universities.head())
# %%
'''
Most difficult majors by AvgDifficulty:
                            AvgRating  AvgDifficulty  NumRatings
MajorField                                                      
Aerospace Eng.  Mechanics         1.8            5.0         5.0
Dentistry                         1.1            4.8         7.0
Epidemiology                      1.9            4.7         7.0
Chemistry  Biochemistry           3.7            4.6         9.0
Disability & Supported Ed.        2.4            4.4        13.0

Most difficult states by AvgDifficulty:
            AvgRating  AvgDifficulty  NumRatings
State                                           
EDINBURGH    2.500000       3.800000        24.0
LANCASHIRE   4.100000       3.800000         5.0
VT           3.892000       3.368000       237.0
MT           3.765000       3.285000       123.0
ME           3.634783       3.243478       178.0

Most difficult universities by AvgDifficulty:
                                    AvgRating  AvgDifficulty  NumRatings
University                                                              
Rockford University                       1.0            5.0         6.0
Bowdoin College                           1.1            5.0         9.0
Northcentral University                   1.4            5.0         8.0
Indiana University at Kokomo              1.0            5.0         9.0
University of Minnesota Law School        1.7            4.8         6.0

'''
#%%
import scipy.stats as stats

# Example: ANOVA for "Tough Grader" tag by MajorField
tag_to_test = 'Tough Grader'

# Group the data by MajorField and extract the tag values
grouped = [group[tag_to_test].dropna().values for name, group in df1.groupby('MajorField') if not group.empty]

# Ensure there is more than one group
if len(grouped) > 1:
    f_stat, p_val = stats.f_oneway(*grouped)
    print(f"ANOVA result for '{tag_to_test}' by MajorField:")
    print(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}")
    if p_val < 0.005:
        print("There is a statistically significant difference in 'Tough Grader' means among Majors.")
    else:
        print("No significant difference found at alpha=0.005.")
else:
    print("Not enough groups to perform ANOVA.")
# %%
'''
ANOVA result for 'Tough Grader' by MajorField:
F-statistic: 2.0571, p-value: 5.3725e-56
There is a statistically significant difference in 'Tough Grader' means among Majors.
'''
from sklearn.cluster import KMeans

# Compute mean tag values by State
state_tag_means = df1.groupby('State')[tag_columns].mean().dropna()

# Perform K-means clustering with, say, 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(state_tag_means)

state_tag_means['Cluster'] = labels
print("States clustered by their average tag profiles:")
print(state_tag_means.sort_values('Cluster'))

# This shows which states fall into which cluster, potentially uncovering patterns in teaching perception.
# %%
'''
States clustered by their average tag profiles:
            Tough Grader  Good Feedback  Respected  Lots to Read  \
State                                                              
SURREY          0.000000       0.000000   0.000000      0.000000   
AB              0.179178       0.240949   0.184979      0.166483   
MT              0.191234       0.288295   0.102942      0.141111   
NB              0.175347       0.201619   0.235595      0.052257   
NC              0.202196       0.247759   0.155733      0.152278   
...                  ...            ...        ...           ...   
London          0.200000       0.200000   0.600000      0.000000   
MANCHESTER      0.000000       0.142857   0.428571      0.000000   
PE              0.045455       0.431818   0.431818      0.083333   
GLASGOW         0.023256       0.162791   0.441860      0.023256   
LANCASHIRE      0.000000       0.600000   0.600000      0.400000   

            Participation Matters  Don’t Skip Class  Lots of Homework  \
State                                                                   
SURREY                   0.400000          0.000000          0.000000   
AB                       0.144649          0.133463          0.084196   
MT                       0.166591          0.228065          0.113106   
NB                       0.069203          0.083608          0.088021   
NC                       0.165697          0.167670          0.152017   
...                           ...               ...               ...   
London                   0.000000          0.000000          0.000000   
MANCHESTER               0.000000          0.000000          0.000000   
PE                       0.083333          0.174242          0.000000   
GLASGOW                  0.093023          0.069767          0.023256   
LANCASHIRE               0.000000          0.000000          0.000000   

            Inspirational  Pop Quizzes  Accessible  ...  Clear Grading  \
State                                               ...                  
SURREY           0.400000     0.000000    0.000000  ...       0.000000   
AB               0.124656     0.023360    0.073056  ...       0.121785   
MT               0.118056     0.026250    0.129210  ...       0.105179   
NB               0.126662     0.008929    0.084251  ...       0.095833   
NC               0.077074     0.037895    0.074953  ...       0.126808   
...                   ...          ...         ...  ...            ...   
London           0.600000     0.000000    0.000000  ...       0.000000   
MANCHESTER       0.571429     0.000000    0.000000  ...       0.000000   
PE               0.378788     0.090909    0.181818  ...       0.000000   
GLASGOW          0.441860     0.000000    0.279070  ...       0.023256   
LANCASHIRE       0.000000     0.000000    0.000000  ...       0.000000   

            Hilarious  Test Heavy  Graded by Few Things  Amazing Lectures  \
State                                                                       
SURREY       0.600000    0.000000              0.000000          0.200000   
AB           0.117369    0.026508              0.034009          0.130397   
MT           0.124365    0.029643              0.011111          0.063434   
NB           0.205202    0.017708              0.006944          0.198645   
NC           0.122562    0.032535              0.039969          0.087354   
...               ...         ...                   ...               ...   
London       0.200000    0.000000              0.000000          0.400000   
MANCHESTER   0.142857    0.000000              0.000000          0.285714   
PE           0.250000    0.136364              0.000000          0.090909   
GLASGOW      0.069767    0.000000              0.000000          0.511628   
LANCASHIRE   0.000000    0.000000              0.000000          0.400000   

              Caring  Extra Credit  Group Projects  Lecture Heavy  Cluster  
State                                                                       
SURREY      0.000000      0.000000        0.000000       0.000000        0  
AB          0.201819      0.016246        0.065995       0.102570        1  
MT          0.133460      0.106710        0.077879       0.050893        1  
NB          0.194949      0.016741        0.078348       0.049206        1  
NC          0.204080      0.086115        0.054274       0.104515        1  
...              ...           ...             ...            ...      ...  
London      0.200000      0.000000        0.000000       0.400000        2  
MANCHESTER  0.000000      0.000000        0.000000       0.000000        2  
PE          0.227273      0.000000        0.090909       0.136364        2  
GLASGOW     0.186047      0.000000        0.000000       0.000000        2  
LANCASHIRE  0.200000      0.200000        0.400000       0.000000        3  

[68 rows x 21 columns]


'''
#%%
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Compute mean tag values by MajorField
major_tag_means = df1.groupby('MajorField')[tag_columns].mean().dropna()

# Apply PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(major_tag_means)

major_tag_means['PC1'] = pcs[:,0]
major_tag_means['PC2'] = pcs[:,1]

# Plot the majors in PC space
plt.figure(figsize=(10,7))
for name, row in major_tag_means.iterrows():
    plt.scatter(row['PC1'], row['PC2'], alpha=0.7)
    # Optionally label each point with the MajorField name
    plt.text(row['PC1']+0.01, row['PC2']+0.01, name, fontsize=8)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Major Fields in PCA Space Based on Tag Distributions')
plt.grid(True)
plt.show()
# %%


# Let's pick a single major and see how tags correlate with difficulty within that major
chosen_major = 'Chemistry'
major_data = df1[df1['MajorField'] == chosen_major]

correlations = {}
for tag in tag_columns:
    if major_data[tag].notnull().sum() > 10:  # ensure enough data
        corr = np.corrcoef(major_data['AvgDifficulty'], major_data[tag])[0,1]
        correlations[tag] = corr

# Sort by absolute correlation
correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
print(f"Correlations between AvgDifficulty and tags for {chosen_major}:")
for t, c in correlations.items():
    print(f"{t}: {c:.4f}")

# This shows which tags most strongly correlate with difficulty within that specific major.
# %%
'''
Correlations between AvgDifficulty and tags for Chemistry:
Tough Grader: 0.6559
Caring: -0.4317
Respected: -0.3898
Good Feedback: -0.3389
Clear Grading: -0.3302
Lots to Read: 0.3214
Hilarious: -0.2783
Don’t Skip Class: 0.2698
Lecture Heavy: 0.2682
Amazing Lectures: -0.2542
Test Heavy: 0.2296
Lots of Homework: 0.2108
Inspirational: -0.2056
Extra Credit: -0.1900
Graded by Few Things: 0.1191
So Many Papers: 0.0825
Pop Quizzes: 0.0745
Accessible: -0.0635
Participation Matters: -0.0531
Group Projects: -0.0103
'''


'''
A11:
From the exploratory analysis conducted, it’s clear that linking qualitative data, such as major field, university, and state information, to the numerical and tag data reveals more nuanced patterns in how students perceive their instructors. For instance, we discovered that certain majors disproportionately feature certain tags. Some fields of study stand out as more frequently associated with “Tough Grader,” while others lean towards tags like “Caring” or “Good Feedback.” This indicates that student perceptions are not uniform across academic disciplines and that the nature of a field may influence how students experience and evaluate their professors.

Further, by applying techniques like clustering and principal component analysis (PCA) to the average tag distributions for states and majors, we uncovered clusters of states and fields that share similar tag usage profiles. This suggests that the cultural or educational environment in different regions or fields may shape how instructors teach and how students respond. These patterns extend beyond simple average ratings and difficulties, showing that context—both geographic and disciplinary—plays a significant role in shaping perceptions captured through these tags.

Finally, analyzing correlations between tags and rating metrics (such as difficulty) within specific majors reveals subtle relationships. For example, in one chosen major, “Tough Grader” strongly correlated with higher difficulty ratings, while “Caring” and “Respected” were inversely related to difficulty. Such findings point to a more complex interplay between instructor qualities and student impressions. Rather than relying solely on numeric averages, these more advanced analyses illuminate how educational context, disciplinary focus, and cultural factors converge to create unique patterns of teaching and learning perceptions.
'''
#%%

# Assume major_stats and most_difficult_majors DataFrames have been created as shown previously.

# Extract a subset of majors of interest
majors_of_interest = ["Aerospace Eng.  Mechanics", "Office Technology", "Consumer Affairs", "Dentistry"]
comparison_df = pd.concat([
    major_stats.loc[major_stats.index.intersection(majors_of_interest)][['AvgRating', 'AvgDifficulty']],
    most_difficult_majors.loc[most_difficult_majors.index.intersection(majors_of_interest)][['AvgRating', 'AvgDifficulty']]
]).drop_duplicates()

# Ensure no duplicates if a major appears in both sets
comparison_df = comparison_df.loc[~comparison_df.index.duplicated()]

# Sort by AvgRating to make the chart more interpretable
comparison_df = comparison_df.sort_values('AvgRating', ascending=False)

# Plot AvgRating and AvgDifficulty side by side
x = range(len(comparison_df))
width = 0.4
fig, ax = plt.subplots(figsize=(8, 6))

ax.bar([i - width/2 for i in x], comparison_df['AvgRating'], width=width, color='skyblue', edgecolor='black', label='AvgRating')
ax.bar([i + width/2 for i in x], comparison_df['AvgDifficulty'], width=width, color='salmon', edgecolor='black', label='AvgDifficulty')

ax.set_xticks(x)
ax.set_xticklabels(comparison_df.index, rotation=45, ha='right')
ax.set_ylabel("Value")
ax.set_title("Comparing Average Ratings and Difficulty Across Selected Majors")
ax.legend()

plt.tight_layout()
plt.show()
# %%
