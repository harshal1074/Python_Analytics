#!/usr/bin/env python
# coding: utf-8

# # Matplot Library

# **1. Import Matplot library and some necessary libraries**

# In[2]:


import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt


# **Generating sin and cos Graph using Matplotlib**

# In[3]:


np.pi


# In[4]:


x = np.linspace(0, np.pi, 10)
sin_y = np.sin(x)
print(x)
print([i for i in sin_y])


# In[5]:


x = np.linspace(0, 4*np.pi, 100)
sin_y = np.sin(x)

plt.plot(x, sin_y)
plt.title('Sine Wave')
plt.show()


# In[6]:


x = np.linspace(0, 4*np.pi, 100)

# Defining wave
sin_y = np.sin(x)
cos_y = np.cos(x)

# Plotting the Wave
plt.plot(x, sin_y, label = 'Sine')
plt.plot(x, cos_y, label = 'Cosine')

# To display label we will have to use legend
plt.legend(loc = 'upper right')    # location should be in small letters

plt.title('Sine Wave and Cosine Wave')

plt.show()


# Legend is used to add labels to the graph given in the plot function. The loc argument takes an integer or string that specifies the location of the legend on the plot. different values for loc, such as lower right, center, upper right, lower left, etc.
# title function adds a title to the plot
# '-' will construct a line like in above plots

# In[7]:


x = np.linspace(0, 4*np.pi, 100)

# Defining wave
sin_y = np.sin(x)
cos_y = np.cos(x)

# Plotting the Wave
plt.plot(x, sin_y, '*')
plt.plot(x, cos_y, '--')

plt.title('Sine Wave and Cosine Wave')

plt.show()


# ## 2. Setting Styles to the Graph

# **Style**

# One way to use styles is use plt.style.use('style_name'). List of available styles is

# In[8]:


print(plt.style.available)


# In[9]:


plt.style.use('grayscale')

plt.plot(x, sin_y, label = 'Sine')
plt.plot(x, cos_y, label = 'Cosine')

plt.title('Sine and Cosine Wave')

plt.legend()
plt.show()


# We can also increase the size of the plot
# 
# plt.figure(figure=(length,height))

# In[10]:


plt.figure(figsize = (18,6))

plt.style.use('grayscale')

plt.plot(x, sin_y, label = 'Sine')
plt.plot(x, cos_y, label = 'Cosine')

plt.title('Sine and Cosine Wave')

plt.legend()
plt.show()


# ## 3. Saving the Plots in our Local Machine

# We have to use plt.figure() and savefig()

# In[11]:


plt.figure(figsize = (20,10))

plt.style.use('seaborn-dark')

plt.plot(x, sin_y, label = 'Sine')
plt.plot(x, cos_y, label = 'Cosine')
plt.legend()

plt.title('Sine and Cosine Wave')

fig = plt.figure()

fig.savefig('Sine and Cosine.png')


plt.show()


# **To display the saved image**

# In[12]:


from IPython.display import Image


# In[13]:


Image('Sine and Cosine.png')


#   ## 4.  Different Types of Plotes

# **a) Line Plots**

# In[14]:


x = np.linspace(0, 4*np.pi, 100)
y = np.sin(x)

plt.style.use('Solarize_Light2')
plt.plot(x, y)

plt.title('Sine and Cosine Wave')

plt.show()


# **Multiple Line Plots**

# In[15]:


x = np.linspace(0, 4*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize = (15,5))
plt.style.use('Solarize_Light2')
plt.plot(x, y_sin, color = 'red')
plt.plot(x, y_cos, color = 'blue')

plt.title('Sine and Cosine Wave')

plt.show()


# In[16]:


x = np.linspace(0, 4*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize = (15,5))
plt.style.use('Solarize_Light2')
plt.plot(x, y_sin, label = 'sine', color = 'red', linewidth = 2)
plt.plot(x, y_cos, label = 'cosine', color = 'blue', linewidth = 3)

plt.legend()

plt.title('Sine and Cosine Wave')

plt.show()


# In[17]:


x = np.linspace(0, 4*np.pi, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize = (15,5))
plt.style.use('Solarize_Light2')
plt.plot(x, y_sin, '*', label = 'sine', color = 'red', linewidth = 4)
plt.plot(x, y_cos, '--', label = 'cosine', color = 'blue', linewidth = 5)

plt.legend()

plt.title('Sine and Cosine Wave')

plt.show()


# **For real world datasets**

# In[18]:


from sklearn.datasets import load_diabetes

dataset = load_diabetes()
dataset


# In[19]:


print(type(dataset))
print(dataset.keys())


# In[20]:


target = dataset['target']
target


# In[21]:


plt.figure(figsize = (15,5))
plt.style.use('fast')

plt.plot(target, color = 'blue', linewidth = 1)

plt.show()


# In[22]:


plt.figure(figsize = (15,5))
plt.style.use('fast')

plt.plot(target[:100] , color = 'blue', linewidth = 1)

plt.show()


# We can also use linestyle command. It has 4 main styles solid, dashed, dashdot and dotted

# **linestyle**

# In[23]:


plt.figure(figsize = (15,5))

plt.plot(x, x+0, linestyle = 'solid', label = 'Solid')
plt.plot(x, x+1, linestyle = 'dashed', label = 'Dashed')
plt.plot(x, x+2, linestyle = 'dashdot', label = 'Dashdot')
plt.plot(x, x+3, linestyle = 'dotted', label = 'Dotted')

plt.legend(loc = 'lower right')

plt.show()


# **Adjusting the Plot Axes**
# 
# Can be done using plt.xlim for x axis
# 
# plt.ylim for y axis

# In[24]:


x = np.linspace(0, 4*np.pi, 100)

#Adjusting Plot Axes
plt.xlim(0,15)
plt.ylim(-2,2)
sin_y = np.sin(x)

plt.plot(x, sin_y)
plt.show()


# In[25]:


## can also use xlim([ , ])
x = np.linspace(0, 4*np.pi, 100)

plt.xlim([0,10])
plt.ylim([-2.5,2.5])

sin_y = np.sin(x)
plt.plot(x, sin_y)
plt.title('Sine Wave')

plt.show()


# **IMPORTANT**
# 
# **set**  
# 
# using .set to combine all these commands

# In[26]:


x = np.linspace(0, 4*np.pi, 100)

fig = plt.axes()

fig.plot(x, np.sin(x))

fig.set(xlim = (0,10), ylim = (-2.5,2.5), xlabel = 'x-axis', ylabel = 'sin(x)', title = 'Sin Wave')

fig.plot()


# **b) scatter plot**

# In[27]:


x = np.random.randn(500)
y = np.random.randn(500)

plt.scatter(x, y, c = 'red', alpha = .8, s = 10)

plt.title('Scatter Plot')

plt.show()


# In this code, the scatter function is used to create a scatter plot with 500 data points 
# 
# c for color, alpha for transparency, s for size

# **Multiple Scatter Plots**

# In[28]:


n = 1000

x1 = np.random.randn(500)
y1 = np.random.randn(500)

x2 = np.random.randn(500) + 2
y2 = np.random.randn(500) + 2

plt.scatter(x1, y1, c = 'red', alpha = .7, s = 10, label = "Group-1")
plt.scatter(x2, y2, c = 'blue', alpha = .7, s = 10, label = "Group-2")

plt.title('Scatter Plot')

plt.show()


# **On Real World Dataset**

# In[29]:


from sklearn.datasets import load_iris

iris = load_iris()
x = iris.data
y = iris.target

plt.scatter(x[:,0], x[:,1], c = 'blue', s = 50, alpha =.8)

plt.xlabel(iris['feature_names'][0])
plt.ylabel(iris['feature_names'][1])

plt.show()


# In[30]:


iris['target_names']


# **marker types**
# 
# marker types Like in Lineplot we had different types of lines here we have different types of markers Can be used as marker='type_of_marker' Some of the markers:'o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd'

# In[31]:


plt.scatter(x[:,0], x[:,1], c = 'blue', s = 50, alpha =.8, marker = 'x')

plt.xlabel(iris['feature_names'][0])
plt.ylabel(iris['feature_names'][1])

plt.show()


# The primary difference between plt.scatter and plt.plot is that plt.scatter gives us the freedom to control the properties of each individual point (size,face_color,edge_color, transparency level etc)

# In[32]:


rng = np.random.RandomState(0) #to get same random number each time

x = rng.randn(100)
y = rng.randn(100)

colors = rng.randn(100)
sizes = rng.randn(100)*1000


#cmap = colormap
plt.scatter(x,y, c = colors, s = sizes, alpha = 0.7, edgecolors = 'black', cmap = 'viridis')


plt.title('Random Scatter Plot')
plt.xlabel('x values')
plt.ylabel('y values')

plt.colorbar()

plt.show()



# **for iris dataset**

# In[33]:


iris = load_iris()

features = iris.data

plt.scatter(features[:,0], features[:,1], alpha = .8, s = features[:,3]*100, c = iris.target, cmap = 'viridis', edgecolor = 'black')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.colorbar()
plt.show()


# **c) Histogram**

# In[34]:


data = np.random.randn(1000)

plt.hist(data)

plt.show()


# In[35]:


data = np.random.randn(10000000)

plt.hist(data, bins = 2000, edgecolor = 'blue', alpha = .8, color = 'blue')

plt.title('Random Histogram')
plt.ylabel('Frequency')
plt.xlabel('Values')

plt.show()


# Individual Bins can be removed using histtype='stepfilled'

# In[46]:


data = np.random.randn(1000)

plt.hist(data, bins = 20, edgecolor = 'Black', alpha = .8, color = 'red')

plt.title('Random Histogram')
plt.ylabel('Frequency')
plt.xlabel('Values')

plt.show()


# In[47]:


data = np.random.randn(1000)

plt.hist(data, bins = 20, edgecolor = 'Black', alpha = .8, color = 'red', histtype = 'stepfilled')
#using stepfilled histtype
plt.title('Random Histogram')
plt.ylabel('Frequency')
plt.xlabel('Values')

plt.show()


# **Multiple Histograms**

# In[58]:


x1 = np.random.randn(1000)
x2 = np.random.randn(1000) + 2
x3 = np.random.randn(1000) + 4

plt.hist(x1, bins = 20, edgecolor = 'black',  alpha = .8)
plt.hist(x2, bins = 20, edgecolor = 'black',  alpha = .8)
plt.hist(x3, bins = 20, edgecolor = 'black',  alpha = .8)

plt.title('Random Histogram')
plt.ylabel('Frequency')
plt.xlabel('Values')

plt.show()


# **2D Histogram**
# 
# hist2D

# In[49]:


x = np.random.randn(1000)
y = np.random.randn(1000)

plt.hist2d(x,y, bins = 20 , cmap = 'viridis')

plt.title('2D Histogram')
plt.ylabel('x')
plt.xlabel('y')

plt.show()


# In[50]:


plt.hist2d(x,y, bins = 20 , cmap = 'Blues')
plt.title('2D Histogram')
plt.ylabel('x')
plt.xlabel('y')
plt.colorbar()

plt.show()


# **d) Pie Charts**

# In[61]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   10  ,  200   , 500      ]

fig = plt.figure(figsize = (7,7))

plt.pie(data,  labels = cars)

plt.show()


# Basic Operations on Pie Chart

# In[54]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   10  ,  200   , 500      ]
explode = [0, 0, 0 , 0, .1, 0] 

fig = plt.figure(figsize = (7,7))

plt.pie(data, labels = cars, explode = explode)

plt.show()


# `Adding Shadows`

# In[63]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   10  ,  200   , 500      ]
explode = [0, 0, 0 , 0, .1, 0] 

fig = plt.figure(figsize = (7,7))

plt.pie(data, labels = cars, explode = explode, shadow = True)
#Adding Shadows

plt.show()


# `Adding Percentage`

# In[71]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   50  ,  200   , 500      ]
explode = [0, 0, 0 , 0, .1, 0] 

fig = plt.figure(figsize = (7,7))

plt.pie(data, labels = cars, explode = explode, shadow = True, autopct = '%2.2f%%')
#Adding Percentage

plt.show()


# **Edge Color**

# In[77]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   50  ,  200   , 500      ]
explode = [0, 0, 0 , 0, .1, 0] 

fig = plt.figure(figsize = (7,7))

plt.pie(data, labels = cars, explode = explode, shadow = True, autopct = '%2.2f%%', wedgeprops = {'edgecolor' : 'black', 'linewidth': True})
#Adding Edge properties

plt.show()


# In[78]:


cars = ['AUDI','BMW','FORD','TESLA','JAGUAR','MERCEDES']
data = [ 450  , 385 , 35   ,   50  ,  200   , 500      ]
explode = [0, 0, 0 , 0, .1, 0] 

fig = plt.figure(figsize = (7,7))

plt.pie(data, labels = cars, explode = explode, shadow = True, autopct = '%2.2f%%', wedgeprops = {'edgecolor' : 'black', 'linewidth' : 2, 'antialiased': True})
#Adding antialiased

plt.show()


# **e) Multiple Subplots**

# In[81]:


x = np.linspace(0,10,50)

fig, ax = plt.subplots(2,2)

y = np.sin(x)
ax[0,0].plot(x,y)
ax[0,0].set_title('Sin Wave')


y = np.cos(x)
ax[0,1].plot(x,y)
ax[0,1].set_title('Cos Wave')

y = np.tan(x)
ax[1,0].plot(x,y)
ax[1,0].set_title('Tan Wave')


y = np.random.randn(1000)
ax[1,1].hist(y,bins = 40)
ax[1,1].set_title('Random Distribution')


plt.tight_layout()
plt.show()


# Subplot Using loop

# In[83]:


plt.subplots_adjust(hspace = .4, wspace = .4)

for i in range(1,7):
    
    plt.subplot(2,3,i)
    plt.text(0.5 , 0.5, str((2,3,i)), fontsize = 18, ha = 'center')
    #0.5, 0.5 is loaction of text to be written


# **Subplot another way to write**

# In[87]:


x = np.linspace(0,10,50)
y = np.sin(x)


fig , (ax1,ax2,ax3) = plt.subplots(1,3, figsize = (18,6))
ax1.plot(x,y)
ax2.plot(x,y**2)
ax3.plot(x,y**3)

ax1.set_title('Sin Wave')
ax2.set_title('Sin Wave - Square')
ax3.set_title('Sin Wave - Cube')

plt.tight_layout()

plt.show()


# **f) 3D Plots**

# In[90]:


plt.axes(projection = '3d')


# In[94]:


ax = plt.axes(projection = '3d')

z = np.linspace(0,15,100)
x = np.sin(z)
y = np.cos(z)

ax.plot3D(x,y,z, 'blue')


# **Another Way**

# In[98]:


from mpl_toolkits.mplot3d import Axes3D

fig  = plt.figure()
ax   = fig.add_subplot(111, projection = '3d')

x = np.random.rand(100)
y = np.random.rand(100)
z = np.random.rand(100)

ax.scatter(x,y,z, c = 'red', marker = '*')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()

plt.show()


# In[ ]:




