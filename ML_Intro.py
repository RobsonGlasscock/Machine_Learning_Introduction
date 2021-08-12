#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary libraries.
from mpl_toolkits import mplot3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sea
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import scipy as scipy

get_ipython().run_line_magic("matplotlib", "inline")

############################# Attributions ##########################
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
# https://pythonprogramming.net/3d-graphing-pandas-matplotlib/
# https://stackoverflow.com/questions/36589521/how-to-surface-plot-3d-plot-from-dataframe
# https://www.python-course.eu/matplotlib_contour_plot.php
# http://www.adeveloperdiary.com/data-science/how-to-visualize-gradient-descent-using-contour-plot-in-python/
# https://www.coursera.org/learn/machine-learning
##################################################


# In[2]:


# import the Stata dataset with the housing data modified to also contain grades and rand.
df = pd.read_stata("house.dta")


# In[3]:


# Look at the first five observations.
df.head(10)


# In[4]:


# Slide 29 shows observations that start with a house with a price of $59,000. 
# Identify these below by identifying all
# all rows with a price that is equal to $59,000.
df[df["price"] == 59000]


# In[5]:


# The data on slide 29  either starts with observation number 17, 23, or 24 above.
df.iloc[17:27]


# In[6]:


# The above output does not line up with slide 29. Next, we look at the 10 observations 
# starting with row the 24th index.
df.iloc[24:34]


# In[7]:


# Slide 31 contains summary statistics in Excel for price. Let's calculate 
# these summary statistics in Python.
# Below is the mean, median, variance, and standard deviation of price.
df["price"].mean(), df["price"].median(), df["price"].var(), df["price"].std()


# In[8]:


# Slide 31, below is the covariance of price and area. 
# The covariance is in the first row, second column.
df[["price", "area"]].cov()


# In[9]:


# Slide 31, below is the correlation of price and area. 
# The correlation is in the first row, second column.
df[["price", "area"]].corr()


# In[10]:


# Slide 32
# Next, let's look at the summary statistics for the dataframe.
df.describe()


# In[11]:


# The format of the output above looks different than Slide 13, but the results are the same.

# Slide 36 Next, create the scatterplot of price and area.
df.plot.scatter(y="price", x="area")


# In[12]:


# Slide 37
ax = sea.regplot(y=df["price"], x=df["area"], ci=None)


# In[13]:


# Silently create model with age and agesq to estimate the nonlinear relationship 
# and then add the predicted values to the dataframe.
X = df[["age", "agesq"]]
X = sm.add_constant(X)
Y = df["price"]
model = sm.OLS(Y, X).fit()
y_hat = model.predict(X)
df["y_hat"] = y_hat


# In[14]:


# Slide 42
fig, ax = plt.subplots()
ax.scatter(y=df["price"], x=df["age"])
ax.scatter(y=df["y_hat"], x=df["age"], color="red")
ax.set_xlabel("age")
ax.set_ylabel("price")
plt.show
del df["y_hat"]


# In[15]:


# Slide 44
df.plot.scatter(y="price", x="grades")


# In[16]:


# Slide 45
ax = sea.regplot(y=df["price"], x=df["grades"], ci=None)


# In[17]:


# Slide 48
df.plot.scatter(y="price", x="rand")


# In[18]:


# Slide 49
ax = sea.regplot(y=df["price"], x=df["rand"], ci=None)


# In[19]:


# Slide 52 - Area
X = df["area"]
X = sm.add_constant(X)
Y = df["price"]
model = sm.OLS(Y, X).fit()
print(model.summary())


# In[20]:


# Slide 52 - Grades
X = df["grades"]
X = sm.add_constant(X)
Y = df["price"]
model = sm.OLS(Y, X).fit()
print(model.summary())


# In[21]:


# Slide 52 - Rand
X = df["rand"]
X = sm.add_constant(X)
Y = df["price"]
model = sm.OLS(Y, X).fit()
print(model.summary())


# In[22]:


# Slides 63 and 68 
X = df["area"]
X = sm.add_constant(X)
Y = df["price"]
model = sm.OLS(Y, X).fit()
print(model.summary())


# In[23]:



# In[24]:


# Slidse 77 and 78.
# Below creates a new variable "scaled" which takes each value of
# area, subtracts the mean of area, and then divides by the standard deviation of area.
df = df.iloc[0:25].copy()
df["scaled"] = (df["area"] - df["area"].mean()) / df["area"].std()
df.head(25)


# In[25]:


# Slides 79 until the end.

# Initial guesses for B0 and B1 are zero.
coeff_guess = np.asarray([[0], [0]], dtype="int64")
# Set the learning rate paramater.
alpha = 0.01 / len(df)

results = []
# The code below does not use linear algebra on purpose.
# This talk was given to accounting and finance folks so I wanted to 
# make the equations below more user friendly for that audience.

# Gradient descent is run with 1,000 iterations below.
for i in range(0, 1000):
    one = (coeff_guess[0, 0] + (coeff_guess[1, 0] * df["scaled"]) - df["price"]).sum()
    two = (
        ((coeff_guess[0, 0] + coeff_guess[1, 0] * df["scaled"]) - df["price"])
        * df["scaled"]
    ).sum()
    derivatives = np.asarray([[one], [two]], dtype="int64")
    # Initial guess.
    if i == 0:
        coeff_guess = np.asarray([[0], [0]], dtype="int64")
        cost = (
            alpha
            * (1 / 2)
            * (
                (coeff_guess[0, 0] + (coeff_guess[1, 0] * df["scaled"]) - df["price"])
                ** 2
            ).sum()
        )
        results.append([i, coeff_guess[0, 0], coeff_guess[1, 0], cost])
    # Guesses for all iterations other than the first one.
    else:
        coeff_guess = coeff_guess - (alpha * derivatives)
        cost = (
            alpha
            * (1 / 2)
            * (
                (coeff_guess[0, 0] + (coeff_guess[1, 0] * df["scaled"]) - df["price"])
                ** 2
            ).sum()
        )
        results.append([i, coeff_guess[0, 0], coeff_guess[1, 0], cost])

results

# Turn the results list into a dataframe.
df2 = pd.DataFrame(results, columns=["iteration", "beta_0", "beta_1", "cost"])

df2["iteration"] = df2["iteration"] + 1
# Below output is comparable to house_gradient.xlsx results. 
# You can see that, within rounding errors, the paramter estimates 
# match.
df2.head(3)
df2.tail(3)

# Create X matrix with a constant. Below lines transform df columns into 
# vectors or matrices and then estimates and OLS model to compare to the 
# gradient descent estimates later.
X_model = sm.add_constant(df["scaled"])
Y = df["price"]

model = sm.OLS(Y, X_model)
output = model.fit()
output.params
print(output.summary())

df2.to_excel("first_three_last_three.xlsx", index=False)

# Below creates a graph of the gradient descent path based on each B0 and B1 
# guess and the cost function with that parameter combination.
fig = plt.figure(figsize=(12, 12))
gd = plt.axes(projection="3d")
plt.yticks(rotation=90)
gd.view_init(20, 10)
gd.set_title("Descent Path")
ax.ticklabel_format(style="plain")
gd.scatter(df2["beta_0"], df2["beta_1"], df2["cost"])
gd.set_xlabel("beta_0", labelpad=13)
gd.set_ylabel("beta_1", labelpad=13)
gd.set_zlabel("cost")
fig.savefig("descent.pdf")

# The graphs make it appear that the constant is less than $50,000. 
# This appears strange, so I am double checking the cost function when 
# B0 is around  $45K- $49K and seeing where the minimum of the cost function is. 
# No exceptions noted. p/f/r.
df2[df2["cost"] == df2["cost"].min()]
df2.iloc[200:300]

# Above we have each combination of B0 and B1, but we don't have the cost 
# function for all combinations of B0 and B1.Next, we write a loop, using 
# linear algebra this time, to "sweep out" the cost function for all combinations 
# of each B0 and B1. The graph created from this will show more than just the 
# gradient descent path because it will show, for example, for each B0 the cost 
# with ALL of the other values of B1 in the range of B1 estimates. 
# You can see this with the nested loops.

SST_Mat = np.zeros((1000, 1000))
X_model = np.asarray(X_model)
Y = np.asarray(Y)
Y = Y.reshape((25, 1))
G, H = np.meshgrid(df2["beta_0"], df2["beta_1"])

for i, j in enumerate(df2["beta_0"]):
    for e, f in enumerate(df2["beta_1"]):
        beta_hat = np.asarray([[j], [f]], dtype="int64")
        XB = np.matmul(X_model, beta_hat)
        Epsilon = np.subtract(XB, Y)
        SST = alpha * (1 / 2) * np.matmul(Epsilon.T, Epsilon)
        SST_Mat[i, e] = SST

fig = plt.figure(figsize=(12, 12))
ax = plt.axes(projection="3d")
plt.yticks(rotation=90)
ax.contour3D(G, H, SST_Mat, 50, cmap="viridis")
ax.set_title("Cost Function")
ax.ticklabel_format(style="plain")
ax.set_xlabel("beta_0", labelpad=13)
ax.set_ylabel("beta_1", labelpad=13)
ax.set_ylabel("beta_1")
ax.set_zlabel("cost")
ax.view_init(20, 10)
fig.savefig("plane.pdf")

# Rg: Note that it may appear that the value of B0 is actually less than 
# $50,000 but that is just because of the rotation of the graph. 
# See below.

fig = plt.figure(figsize=(11, 11))
ax = plt.axes(projection="3d")
plt.yticks(rotation=20)
ax.contour3D(G, H, SST_Mat, 50, cmap="viridis")
ax.set_title("Cost Function")
ax.ticklabel_format(style="plain")
ax.set_xlabel("beta_0", labelpad=13)
ax.set_ylabel("beta_1", labelpad=13)
ax.set_ylabel("beta_1")
ax.set_zlabel("cost")
ax.view_init(45, 45)


# In[ ]:
