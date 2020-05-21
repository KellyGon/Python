# -*- coding: utf-8 -*-
"""
DataCamp course - Basic and Intermediate
@author: Kelly
"""
#
# Command em same line:
#command1; command2

# =============================================================================
# #[1] VARIABLES
# =============================================================================

#[1.1] Creating a variable

height=1.79
weight = 74.2
bmi= weight/height*2

#type of the variable
type(bmi)

#[1.2] Create a variable boolen
boolean=True

#[1.3] Operators: +,-,*,/,%
#Double stars (**) are exponentiation
investment= 100 * 1.1 ** 7 #

#[1.4] Convert variables

#convert a variable into string
print("I started with $" + str(investment))

# =============================================================================
# #[2] LISTS
# =============================================================================

#[2.1] Define a vector
vec=[1.71,1.32]
vec2=["A",1.71]
vec3=[["A",1.71],["B",1.32]]
vec4=[1 + 2, "a" * 5, 3]

#[2.2] Proprieties of index of the vectors
#Vector starts to count from 0
vec[0]
vec3[1]
x = ["a", "b", "c", "d"]
x[1:3] #last index don't count
#from 0 to 2
x[:2]
#from 3 to the end
x[2:]
#all
x[:]

#[2.2] Index of vectors 2D
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
x[2][0]
x[2][:2]

#sum of vectors
vec[0]+vec[1]

#delete
del(x[2])

#replace
x[1] = "r"
x[2:] = ["s", "t"]
x[:]

#coping
y=list(x)
y=x[:] #não pode por igual pois copia apenas as referencias

# =============================================================================
# #[3] FUNCTIONS
# =============================================================================

round(1.68,1)
round(1.68)
help(round)

#[3.1] Methods (funcao de um determiado objeto)
x.index(["a", "b", "c"])
x.append("z")
x

name="Liz"
name.replace("L","l")
name.capitalize()

#[3.2] Packages 
import math
math.pi

from scipy.linalg import inv as my_inv
my_inv([[1,2], [3,4]])

# =============================================================================
# #[4] NUMPY
# =============================================================================

#[4.1] Importing numpy
import numpy
numpy.array([1,2,3])

import numpy as np
np.array([1,2,3])

from numpy import array
array([1,2,3])

#lists converted to arrays - arrays cannot contain elements with different types.
h=[1.73,1.68]
w=[60,50]
np_h=np.array(h)
np_w=np.array(w)
bmi=np_w/np_h**2

h=array([1.73,1.68])

bmi>18

#[4.2] Lists X Arrays
#sum of lists=[1, 2, 1, 2]
[1,2]+[1,2]
#sum of arrays= array([2, 4])
array([1,2])+array([1,2])

#index of lists
x = [["a", "b"], ["c", "d"]]
[x[0][0], x[1][0]] #[a,c]
#index of arrays
np_x = np.array(x)
np_x[:,0]  #[a,c]

#[4.3] Boolean
y = np.array([4 , 9 , 6, 3, 1])
high = y > 5
y[high] #[9,6]

#[4.4] Arrays store only one type of objects-> True=1, False=0
np.array([True, 1, 2]) + np.array([3, 4, False]) #array([4, 5, 2])

#[4.5] 2D arrays 
np_2d=np.array([[1,2,3],[4,5,6]])
np_2d.shape
np_2d[0]
np_2d[0][2]
np_2d[1,:] #segunda linha, todas as colunas
np_2d[:,1] #segunda coluna, todas as linhas
np_2d[1,0:2] #segunda linha, coluna 0,1

np_mat = np.array([[1, 2],
                   [3, 4],
                   [5, 6]])
np_mat * 2
np_mat + np.array([10, 10])
np_mat + np_mat

# =============================================================================
# #[5] Basic Statistics
# =============================================================================

np_h=np.round(np.random.normal(1.75,0.2,5000))
np.mean(np_h[:])
np.mean(np_h)
np.mean(np_h[0:2500])

# =============================================================================
# #[6] PLOTS
# =============================================================================

#[6.1] Plot(x,y)
import matplotlib.pyplot as plt
year=[1950,1960,1970,1980,1990]
pop=[2,3,5,6,7]
plt.plot(year,pop)
plt.xscale('log')
plt.show()
plt.scatter(year,pop)
plt.show()

#[6.2] Histogram
values=[1,2,2,3,4,5,6,6,6,7,8,9,10]
plt.hist(values,bins=5)

#[6.3] Customiation
# Import numpy as np
import numpy as np

# Store pop as a numpy array: np_pop
np_pop=np.array(pop)

# Double np_pop
np_pop=np_pop*2

# Update: set s argument to np_pop
plt.scatter(year, pop, s = np_pop)

# Previous customizations
plt.xscale('log') 
plt.xlabel('GDP per Capita [in USD]')
plt.ylabel('Life Expectancy [in years]')
plt.title('World Development in 2007')
plt.xticks([1000, 10000, 100000],['1k', '10k', '100k'])

# Add grid() call
plt.grid(True)
plt.show()

# =============================================================================
# #[7] DISCTIONARIES
# =============================================================================

#Precisamos de indexes para conectar duas listas
pop=[30,2,39]
countries=['alfeganistao','albania','algeria']
pop[countries.index('albania')]

#fazer um dicionario e melhor:
world={'alfeganistao':30,'albania':2,'algeria':39}
world['albania']

#[7.1] Keys - header of dic
print(world.keys())

#boolen
'albania' in world 
#add
world['italy']=10
#replace
world['albania']=3
#del
del(world['albania'])

#[7.2] Dic with two keys + add new information frm another dic
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['spain']['capital'])

# Create sub-dictionary data
data={'capital':'rome','population':59.83}

# Add data to europe under key 'italy'
europe['italy']=data

# Print europe
print(europe)

# =============================================================================
# #[8] PANDAS (datasets in python)
# =============================================================================

import pandas as pd

#[8.1] Creating a dataFrame from a dic
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

my_dict={'country':names, 'drives_right': dr,'cars_per_cap':cpc}
cars=pd.DataFrame(my_dict)
print(cars)

#Creating a dataFrame from a csv
brics=pd.read_csv("path/brics.csv", index_col=0) #index_col= 0, so that the first column is used as row labels.

#[8.2] Changing index of a dataFrame
# Build cars DataFrame
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]
cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
cars = pd.DataFrame(cars_dict)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index=row_labels

# Print cars again
print(cars)

#[8.3] Column/Row Access
#column
brics[["country"]]
#Rows
brics[1:4]

#[8.4] Row Acces loc
brics.loc[["RU"]] #entire line
brics.loc[["RU"],["country"]] #only column country of RU
brics.loc[:,["country"]] #all lines of column country

#[8.5] Row Acces iloc (index) RU - index 1, country-0
brics.iloc[[1]] #=brics.loc[["RU"]]
brics.iloc[:,[0]] #=brics.loc[:,["country"]]

# =============================================================================
# #[9] LOOP
# =============================================================================
#[9.1] Numeric comparisons <,>,<=,>=,==,!= and Logical opertators
"carl"<"chris" #True - ordem alfabetca
# Compare a boolean with an integer
True==1 #true
# Comparison of booleans
True>False #true-> 1>0

#AND
True and True #True
False and True #False
False and False #False

#OR
True or True #True
False or True #True
False or False #False

#NOT
not True #false
not False #true

# Define variables
my_kitchen = 18.0
your_kitchen = 14.0

# my_kitchen bigger than 10 and smaller than 18?
my_kitchen > 10 and my_kitchen < 18

# my_kitchen smaller than 14 or bigger than 17?
my_kitchen<14 or my_kitchen>17 

# Double my_kitchen smaller than triple your_kitchen?
2*my_kitchen<3*your_kitchen

#Arrays with logical operators
np.logical_and(bmi>19, bmi<22)

#[9.2] If, elif, else
area = 10.0
if(area < 9) :
    print("small")
elif(area < 12) :
    print("medium")
else :
    print("large")
    
#[9.3] Filtering Pandas DataFrames
#Subseting   
cars = pd.read_csv('cars.csv', index_col = 0)
dr=cars['drives_right']
sel=cars[dr]

#or
sel=cars[cars['drives_right']]
#with logical operator:
cars[np.logical_and(cars['drives_right']>100, cars['drives_right']<500)]

#[9.4] While
error = 50.0
while error > 1 :
    error = error / 4
    print(error)
    
#[9.5] For
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for i in areas : 
    print(i)

# Change for loop to use enumerate() and update print()
for index, area in enumerate(areas) :
    print("room" + str(index+1)+ ":" + str(area))
    
# For with lists of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for i in house:
   print("the " + i[0] + " is " + str(i[1]) +" sqm")
   
#[9.6] For with dictionaries and arrays
   
#Dic - Method
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin',
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'austria':'vienna' }
          
# Iterate over europe
for key, value in europe.items():
    print("the capital of " + key + " is " + value)
    
#Array - Functions
np_areas=np.array(areas)
# For loop over np_height
for x in np.nditer(np_areas):
    print(str(x) + " inches")
    
# Import Twitter data as DataFrame: df
df = pd.read_csv('tweets.csv')

# Initialize an empty dictionary: langs_count
langs_count = {}

# Extract column from DataFrame: col
col = df['lang']

# Iterate over lang column in DataFrame
for entry in col:

    # If the language is in langs_count, add 1 
    if entry in langs_count.keys():
        langs_count[entry]=langs_count[entry]+1
    # Else add the language to langs_count, set the value to 1
    else:
        langs_count[entry]=1

#[9.7] For with DataFrame (nao recomendado)
# Import cars data
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Iterate over rows of cars
for x, row in cars.iterrows():
    print(x) #label of row
    print(row) #row
    
# Adapt for loop
for lab, row in cars.iterrows() :
    print(str(lab) + ": " + str(row['cars_per_cap']))   #apenas coluna per capita
    
#Creating
# Code for loop that adds COUNTRY column
for lab, row in cars.iterrows() :
    cars.loc[lab, "COUNTRY"] = row["country"].upper()
    
#Same thing:
cars["COUNTRY"] = cars["country"].apply(str.upper)

# =============================================================================
# #[10] RANDOM NUMBERS
# =============================================================================

import numpy as np

#[10.1] Random
np.random.rand() #[0,1]
np.random.seed(123) #sets the random seed, so that your results are reproducible between simulations. 
coin=np.random.randint(0,2)

#[10.2] Randon walk
# Initialize random_walk
random_walk=[0]

for x in range(100) :
    # Set step: last element in random_walk
    step=random_walk[-1]

    # Roll the dice
    dice = np.random.randint(1,7)

    # Determine next step
    if dice <= 2:
        step = step - 1
    elif dice <= 5:
        step = step + 1
    else:
        step = step + np.random.randint(1,7)

    # append next_step to random_walk
    random_walk.append(step)
    
#[10.3] Distribution
# Initialize all_walks (don't change this line)
all_walks = []

# Simulate random walk 10 times
for i in range(10) :

    # Code from before
    random_walk = [0]
    for x in range(100) :
        step = random_walk[-1]
        dice = np.random.randint(1,7)

        if dice <= 2:
            step = max(0, step - 1)
        elif dice <= 5:
            step = step + 1
        else:
            step = step + np.random.randint(1,7)
        random_walk.append(step)

    # Append random_walk to all_walks
    all_walks.append(random_walk)

# Print all_walks
print(all_walks)

# =============================================================================
# #[11] Built-in functions
# =============================================================================

def square():
    new_value = 4 ** 2
    return new_value

square()

def square(value):
    new_value =value ** 2
    return new_value

square(4)

#[11.1] Functions with multiple values
def shout_all(word1, word2):
    
    # Concatenate word1 with '!!!': shout1
    shout1=word1+'!!!'
    
    # Concatenate word2 with '!!!': shout2
    shout2=word2+'!!!'
    
    # Construct a tuple with shout1 and shout2: shout_words
    shout_words=(shout1,shout2)

    # Return shout_words
    return shout_words

# Pass 'congratulations' and 'you' to shout_all(): yell1, yell2
(yell1, yell2)=shout_all('congratulations','you')

#[11.2] Using dic
# Define count_entries()
def count_entries(df, col_name):
    """Return a dictionary with counts of 
    occurrences as value for each key."""

    # Initialize an empty dictionary: langs_count
    langs_count = {}
    
    # Extract column from DataFrame: col
    col = df[col_name]
    
    # Iterate over lang column in DataFrame
    for entry in col:

        # If the language is in langs_count, add 1
        if entry in langs_count.keys():
            langs_count[entry]= langs_count[entry]+1
        # Else add the language to langs_count, set the value to 1
        else:
             langs_count[entry]=1

    # Return the langs_count dictionary
    return langs_count

# Call count_entries(): result
lang=['en','esp','en','en','esp','en','esp']
tweets = { 'lang':lang, 'drives_right':dr, 'cars_per_cap':cpc }
tweets_df = pd.DataFrame(tweets)
result=count_entries(tweets_df,'lang') #4 en e 3 esp

# =============================================================================
# #[12] Scope and user-defined functions
# =============================================================================

#Global: main body of script
#Local: variables defined inside a function

def func1():
    num = 3
    print(num)

def func2():
    global num
    double_num = num * 2
    num = 6
    print(double_num)

# Create a string: team
team = "teen titans"

# Define change_team()
def change_team():

    global team #mudou a variavel globalmente - a var que esta fora da função
    team="justice league"
    print(team)

change_team()
print(team)

#Nested functions
# Define three_shouts
def three_shouts(word1, word2, word3):
    """Returns a tuple of strings
    concatenated with '!!!'."""

    # Define inner
    def inner(word):
        """Returns a string concatenated with '!!!'."""
        return word + '!!!'

    # Return a tuple of strings
    return (inner(word1), inner(word2), inner(word3))

# Call three_shouts() and print
print(three_shouts('a', 'b', 'c'))

# Define echo_shout()
def echo_shout(word):
    """Change the value of a nonlocal variable"""
    
    # Concatenate word with itself: echo_word
    echo_word=word+word
    
    # Print echo_word
    print(echo_word)
    
    # Define inner function shout()
    def shout():
        """Alter a variable in the enclosing scope"""    
        # Use echo_word in nonlocal scope
        nonlocal echo_word
        
        # Change echo_word to echo_word concatenated with '!!!'
        echo_word = echo_word+'!!!'
    
    # Call function shout()
    shout()
    
    # Print echo_word
    print(echo_word)

# Call function echo_shout() with argument 'hello'
echo_shout('hello')

#[12.1] Add a default argument
# Define shout_echo
def shout_echo(word1,echo=1):

    echo_word = echo*word1
    return echo_word

no_echo = shout_echo("Hey")
with_echo = shout_echo("Hey",5) 

#[12.2] Flexible arguments
# Define shout_echo
def shout_echo(word1, echo=1, intense=False):
    echo_word = word1 * echo
    if intense is True:
        # Make uppercase and concatenate '!!!': echo_word_new
        echo_word_new = echo_word.upper() + '!!!'
    else:
        # Concatenate '!!!' to echo_word: echo_word_new
        echo_word_new = echo_word + '!!!'
    return echo_word_new

# Call shout_echo() with "Hey", echo=5 and intense=True: with_big_echo
with_big_echo = shout_echo("Hey",5,True)

# Call shout_echo() with "Hey" and intense=True: big_no_echo
big_no_echo = shout_echo("Hey", intense=True)

# Print values
print(with_big_echo)
print(big_no_echo)


#[12.3] Define a flexible argument
def gibberish(*args):
    hodgepodge=""
    for word in args:
        hodgepodge += word
    return hodgepodge

# Call gibberish() with one string: one_word
one_word = gibberish("luke")

# Call gibberish() with five strings: many_words
many_words = gibberish("luke", "leia", "han", "obi", "darth")

# Print one_word and many_words
print(one_word)
print(many_words)

#Another example:
# Define count_entries()
def count_entries(df, *args):
    cols_count = {}
    
    # Iterate over column names in args
    for col_name in args:
        col = df[col_name]
        for entry in col:
            if entry in cols_count.keys():
                cols_count[entry] += 1
            else:
                cols_count[entry] = 1
    return cols_count

# Call count_entries(): result1
result1 = count_entries(tweets_df, 'lang')

# Call count_entries(): result2
result2 = count_entries(tweets_df, 'lang', 'source')

#[12.4] Define a dictionary as argument
def report_status(**kwargs):
    print("\nBEGIN: REPORT\n")

    for key, v in kwargs.items():
        # Print out the keys and values, separated by a colon ':'
        print(key + ": " + v)

    print("\nEND REPORT")

report_status(name="luke",affiliation="jedi",status="missing")


# =============================================================================
# [13] Lambda function
# =============================================================================
#Defining a function in one line
add_bangs = (lambda a: a + '!!!')
add_bangs('hello')

# Defining and calling a function in one line
spells = ["protego", "accio", "expecto patronum", "legilimens"]
shout_spells = map(lambda a:a+"!!!", spells)
shout_spells_list=list(shout_spells)
print(shout_spells_list)

# Select retweets from the Twitter DataFrame: result
result = filter(lambda x: x[0:2]=='RT', tweets_df['text'])
res_list=list(result)
for tweet in res_list:
    print(tweet)
    
# =============================================================================
# [14] Introduction to error handling
# =============================================================================
#[14.1] Try-except

# Define shout_echo
def shout_echo(word1, echo=1):
     echo_word=""
     shout_words=""
     
    try:
        echo_word = echo*word1
        shout_words = echo_word + '!!!'
    except:
        print("word1 must be a string and echo must be an integer.")
    return shout_words

# Call shout_echo
shout_echo("particle", echo="accelerator")

#[14.2] Raise TypeError
# Define shout_echo
def shout_echo(word1, echo=1):
    if echo<0:
        raise ValueError('echo must be greater than or equal to 0')

    echo_word = word1 * echo
    shout_word = echo_word + '!!!'
    return shout_word

# =============================================================================
# [15] Interators
# =============================================================================

#[15.1]  For x Interator
# Create a list of strings: flash
flash = ['jay garrick', 'barry allen', 'wally west', 'bart allen']

# Print each list item in flash using a for loop
for person in flash:
    print(person)
    
# Create an iterator for flash: superhero
superhero=iter(flash)

# Print each item from the iterator
print(next(superhero))
print(next(superhero))
print(next(superhero))
print(next(superhero))

#[15.2] Iter with range
# Create an iterator for range(3): small_value
small_value = iter(range(3))
print(next(small_value))
print(next(small_value))
print(next(small_value))

# Create a range object: values
values = range(10,21)
values_sum = sum(values)
print(values_sum)

#[15.3] Enumerate
#Transforma uma lista em uma lista enumerada
mutants = ['charles xavier', 
            'bobby drake', 
            'kurt wagner', 
            'max eisenhardt', 
            'kitty pryde']
mutant_list = list(enumerate(mutants))
print(mutant_list)

# Unpack and print the tuple pairs
for index1, value1 in enumerate(mutants):
    print(index1, value1)
# Change the start index
for index2, value2 in enumerate(mutants,1):
    print(index2, value2)


#[15.4] Zip
aliases=['prof x', 'iceman', 'nightcrawler', 'magneto', 'shadowcat']
powers=['telepathy',
 'thermokinesis',
 'teleportation',
 'magnetokinesis',
 'intangibility']

# Create a list of tuples: mutant_data
mutant_data = list(zip(mutants,aliases,powers))
print(mutant_data)

# Unpack the zip object and print the tuple values
mutant_zip = zip(mutants,aliases,powers)
print(mutant_zip)
for (value1, value2, value3) in mutant_zip:
    print(value1, value2, value3)
    
# 'Unzip' with *
result1, result2, result3 = zip(*mutant_zip)
print(result1)
print(result2)
print(result3)

#[15.5] Chunks - abrir a base por partes
counts_dict={}
for chunk in pd.read_csv('tweets.csv',chunksize=10) :
    for entry in chunk['lang']:
        if entry in counts_dict.keys():
            counts_dict[entry] += 1
        else:
            counts_dict[entry] = 1

print(next(counts_dict))   
         
# =============================================================================
# [16] List comprehensions
# =============================================================================

#[16.1] Creating lists with for
doctor = ['house', 'cuddy', 'chase', 'thirteen', 'wilson']
[doc[0] for doc in doctor]

# Create list comprehension: squares
[i**2 for i in range(0,10)]

#Creating a matrix
# Create a 5 x 5 matrix using a list of lists: matrix
matrix = [[col for col in range(5)] for row in range(5)]
for row in matrix:
    print(row)
    
#[16.2]  Conditional List comprehensions
# Create a list of strings: fellowship
fellowship = ['frodo', 'samwise', 'merry', 'aragorn', 'legolas', 'boromir', 'gimli']
new_fellowship = [member for member in fellowship if len(member)>=7]
print(new_fellowship)

#if-else
new_fellowship = [member if len(member) >= 7 else '' for member in fellowship]

#[16.3]  Dictionary List comprehensions
new_fellowship = {member:len(member) for member in fellowship }
print(new_fellowship)  

#[16.4] Generators
# List of strings
fellow2 = (member for member in fellowship if len(member) >= 7)
print(fellow2)  #generated a list

#With functions
def get_lengths(input_list):
    for person in input_list:
        yield len(person)
for value in get_lengths(fellowship):
    print(value)
    
#[16.5] Index in List comprehensions     
[i[2:4] for i in fellowship]
[i[2:4] for i in fellowship if i[2]=='r']

# =============================================================================
# [17] Case of Study - World Bank
# =============================================================================

import pandas as pd

df=pd.read_csv(r"C:\Users\Kelly\OneDrive - Fundacao Getulio Vargas - FGV\Doutorado\Curso Python - Data Camp\Python Data Science Toolbox/WDIData.csv", sep=',' )
feature_names = list(df['Country Name'])
row_vals = list(df['2017'])

# Define lists2dict()
def lists2dict(list1, list2):
    zipped_lists = zip(list1, list2)
    rs_dict = dict(zipped_lists)
    return rs_dict

rs_fxn = lists2dict(feature_names,row_vals)
list_of_dicts = [lists2dict(feature_names,sublist) for sublist in df]
list_of_dicts[0]

#Set a dic as a dataFrame
df2 = pd.DataFrame(list_of_dicts)
print(df.head())

#Creating a dic from csv and counting values for each country
# Open a connection to the file
with open(r"C:\Users\Kelly\OneDrive - Fundacao Getulio Vargas - FGV\Doutorado\Curso Python - Data Camp\Python Data Science Toolbox/WDIData.csv") as file:
    file.readline()
    counts_dict = {}
    #count the first 15.000 rows
    for j in range(15000):
        line = file.readline().split(',')
        first_col = line[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1

print(counts_dict)

#Counting all rows
#Define a generator
def read_large_file(file_object):
    while True:
        data = file_object.readline()
        if not data:
            break
        yield data

#Open        
counts_dict = {}
with open(r"C:\Users\Kelly\OneDrive - Fundacao Getulio Vargas - FGV\Doutorado\Curso Python - Data Camp\Python Data Science Toolbox/WDIData.csv") as file:
    for line in read_large_file(file):
        row = line.split(',')
        first_col = row[0]
        if first_col in counts_dict.keys():
            counts_dict[first_col] += 1
        else:
            counts_dict[first_col] = 1


# Columns of interest
df_CEB = df[df['Country Code']=='CEB']
indicators = zip(df_CEB['2017'], df_CEB['2018'])
ind_list = list(indicators)
print(ind_list)


#########################Final Exercise 
# Initialize reader object: urb_pop_reader
# Define plot_pop()
def plot_pop(filename, country_code):

    data = pd.DataFrame()
    df_pop_ceb = filename[filename['Country Code'] == country_code]
    pops = zip(df_pop_ceb['2017'],
                df_pop_ceb['2018'])
    pops_list = list(pops)
    df_pop_ceb['Difference'] = [int(tup[0] - tup[1]) for tup in pops_list]
    data = data.append(df_pop_ceb)
    data.plot(x='Difference')
    plt.show()

# Set the filename: fn
df=df[['Country Code','2017','2018']]
df = df.dropna()

# Call plot_pop for country code 'CEB'
plot_pop(df,'CEB')

# Call plot_pop for country code 'ARB'
plot_pop(fn,'ARB')
