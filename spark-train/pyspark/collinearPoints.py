
# coding: utf-8

# # Programming Assignment 1: Collinear Points
# 
# For this programming assignment, we'll be using a Jupyter notebook.

# ## Background

# ### Collinear points
# 
# Definition of collinearity[1]: In geometry, collinearity of a set of points is the property of their lying on a single line. A set of points with this property is said to be collinear.
# 
# ![](non-collinear-points.jpg)
# 
# Here, points P,Q,R and A,R,B are collinear. However, points A,B,C are non-collinear. For more, refer [2].
# 
# 1. https://en.wikipedia.org/wiki/Collinearity
# 2. http://www.mathcaptain.com/geometry/collinear-points.html

# ### Parameterizing lines
# In order to determine whether a set of points all lie on the same line we need a standard way to define (or parametrize) a line.
# 
# * One way of defining a line is as the set of points $(x,y)$ such that $y=ax+b$ for some fixed real values $a,b$.
# * We call $a$ the **slope** of the line and $b$ is the $y$-intercept which is defined as the value of $y$ when $x=0$.
# * This parameterization works for *almost* all lines. It does not work for vertical lines. For those lines we define $a$ to be **infinity** and $b$ to be the $x$ intercept of the line (the line is parallel to the $y$ axis so it does not intercept the $y$ axis (other than if it is the vertical line going through the origin).
# 
# To summarize, given two different points $(x_1,y_1) \neq (x_2,y_2)$, we define the parameterization $(a,b)$ as:
# * **if $x_1=x_2$: ** $(\mbox{Inf},x_1)$ 
# * **Else:** $(a,b)$ such that $y_1=a x_1 +b$ and $y_2=a x_2 +b$.
# 

# ## Task
# 
# Given an input file with an arbitrary set of co-ordinates, your task is to use pyspark library functions and write a program in python3 to find if three or more points are collinear.
# 
# For instance, if given these points: {(1,1), (0,1), (2,2), (3,3), (0,5), (3,4), (5,6), (0,-3), (-2,-2)}
# 
# Sets of collinear points are: {((-2,-2), (1,1), (2,2), (3,3)), ((0,1), (3,4), (5,6)), ((0,-3), (0,1), (0,5))}. Note that the ordering of the points in a set or the order of the sets does not matter. 
# 
# Note: 
# <ul>
#   <li>Every set of collinear points has to have <b>at least three points</b> (any pair of points lie on a line).</li>
#   <li>There are two types of test cases:
#       <ul>
#       <li><b>Visible Test cases</b>: Test cases given to you as a part of the notebook. These tests will help you validate your program and figure out bugs in it if any.</li>
#       <li><b>Hidden Test cases</b>: Test cases that are not given as a part of the notebook, but will be used for grading. <br>Cells in this notebook that have "<i>##Hidden test cases here</i>" are read-only cells containing hidden tests.</li>
#       </ul>
#   </li>
#   <li>Any cell that does not require you to submit code cannot be modified. For example: Assert statement unit test cells. Cells that have "**# YOUR CODE HERE**" are the ONLY ones you will need to alter. </li>
#   <li>DO NOT change the names of functions. </li>
#       
# </ul>

# ### Description of the Approach
# 
# The goal of this assignment is to make you familiar with programming using pyspark. There are many ways to find sets of collinear points from a list of points. For the purposes of this assignment, we shall stick with the below approach:
# 
# 1. List all pairs of points. You can do that efficiently in spark by computing cartesian product of the list of points with itself. For example, given three points $[(1,0), (2,0), (3,0)]$, we construct a list of nine pairs  
# $[((1,0),(1,0)),((1,0), (2,0)),((1,0),(3,0))$  
# $((2,0),(1,0)),((2,0), (2,0)),((2,0),(3,0))$  
# $((3,0),(1,0)),((3,0), (2,0)),((3,0),(3,0))]$  
# 
# 2. Remove the pairs in which the same point appears twice such as $((2,0),(2,0))$. After these elimination you end up (for this example) with a list of just six pairs:  
# $[((1,0),(2,0)),((1,0),(3,0)),((2,0),(1,0)),((2,0),(3,0)),((3,0),(1,0)),((3,0),(2,0))]$
# 
# 2. For each pair of points, find the parameterization $(a,b)$ of the line connecting them as described above.
# 
# 3. Group the pairs according to their parameters. Clearly, if two pairs have the same $(a,b)$ values, all points in the two pairs lie on the same line.
# 
# 3. Eliminate the groups that contain only one pair (any pair of points defines a line).
# 4. In each of the remaining groups, unpack the point-pairs to identify the individual points.
# Note that if a set of points $(x_1,y_1),\ldots,(x_k,y_k)$ lie on the same line then each point will appear $k-1$ times in the list of point-pairs. You therefore need to transform the list of points into sets to remove duplicates.
# 
# 5. Output the sets of 3 or more colinear points.
# 
# Your task is to implement the described algorithm in Spark. You should use RDD's all the way through and collect the results into the driver only at the end.

# ### Notebook Setup

# In[2]:


import os

os.environ["PYSPARK_PYTHON"]="/usr/bin/python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/usr/bin/python3"

from pyspark import SparkContext, SparkConf

#We can create a SparkConf() object and use it to initialize the spark context
conf = SparkConf().setAppName("Collinear Points").setMaster("local[4]") #Initialize spark context using 4 local cores as workers
sc = SparkContext(conf=conf)    

from pyspark.rdd import RDD


# ### Helper Functions
# Here are some helper functions that you are encouraged to use in your implementations. Do not change these functions.

# The function <font color="blue">format_result</font> takes an element of the form shown below in the example. It outputs a tuple of all points that are collinear (shown below).
# 
# Input: ((A,slope), [C1,..., Ck]) where each of A, C1, ..., Ck is a point of form (Ax, Ay) and slope is of type float.
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# my_input = (((2, 1), 0.5), [(4, 2), (6, 3)]) 
# format_result(my_input)
# ```
# Output: (C1,..., Ck, A) each of A,C1,...,Ck is a point of form (Ax, Ay)
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# ((4, 2), (6, 3), (2, 1))
# ```
# 
# <font color="red">**Hint : **</font> The above example is given just to provide the input and output format. This function is called a different way in the spark exercise.
# 

# In[3]:


def format_result(x):
    x[1].append(x[0][0])
    return tuple(x[1])


# In[4]:


def to_sorted_points(x):
    """
    Sorts and returns a tuple of points for further processing.
    """
    return tuple(sorted(x))


# ## Exercises
# 
# Here are some functions that you will implement. You should follow the function definitions, and use them appropriately elsewhere in the notebook.

# ### Exercise 1: to_tuple

# #### Example
# The function <font color="blue">to_tuple</font> converts each point of form 'Ax Ay' into a point of form (Ax, Ay) for further processing.
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# my_input = '2 3'
# to_tuple(my_input)
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# (2, 3)
# ```
# 
# <font color="red">**Hint : **</font> The above example is given just to provide the input and output format. This function is called a different way in the spark exercise.
# 

# #### Definition

# In[147]:


## Insert your answer in this cell. DO NOT CHANGE THE NAME OF THE FUNCTION.
def to_tuple(x):
    ###
    ### YOUR CODE HERE
    ###
    xl=x.split()
    return (int(xl[0]),int(xl[1]))


# #### Unit Tests

# In[6]:


assert type(to_tuple('1 1')) == tuple, "Incorrect type: Element returned is not a tuple"


# In[7]:


assert type(to_tuple('1 1')[0])==int and type(to_tuple('1 1')[1])==int, "Incorrect element type: Element returned is not an integer"


# In[8]:


assert to_tuple('1 1') == (1,1), "Incorrect Return Value: Value obtained does not match"


# ### Exercise 2: non_duplicates

# #### Example
# 
# The function <font color="blue">non_duplicates</font> checks if a set of points contains duplicates or not.
# 
# Input: Pair (A,B) where A and B are of form (Ax, Ay) and (Bx, By) respectively.
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# my_input = ((0,0),(1,2))
# non_duplicates(my_input)
# ```
# 
# Output: Returns True if A != B, False otherwise.
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# True
# ```
# 
# <font color="red">**Hint : **</font> The above example is given just to provide the input and output format. This function may be used to "filter" out duplicates inside the get_cartesian() function.

# #### Definition

# In[9]:


## Insert your answer in this cell. DO NOT CHANGE THE NAME OF THE FUNCTION.
def non_duplicates(x):
    """ 
    Use this function inside the get_cartesian() function to 'filter' out pairs with duplicate points
    """
    ###
    ### YOUR CODE HERE
    ###
    test =False
    if (x[0] != x[1]):
        test=True
    return test


# #### Unit Tests

# In[10]:


assert type(non_duplicates(((0,0),(1,2)))) == bool, "Incorrect Return type: Function should return a boolean value"


# In[11]:


assert non_duplicates(((0,0),(1,2))) == True, "No duplicates are present"


# In[12]:


assert non_duplicates(((0,0),(0,0))) == False, "Duplicates exist: (0,0)"


# ### Exercise 3: get_cartesian

# #### Example
# 
# The function <font color="blue">get_cartesian</font> does a cartesian product of an RDD with itself and returns an RDD with <b>DISTINCT</b> pairs of points.
# 
# Input: An RDD containing the given list of points
# 
# Output: An RDD containing The cartesian product of the RDD with itself
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# test_rdd = sc.parallelize([(1,0), (2,0), (3,0)])
# get_cartesian(test_rdd).collect()
# ```
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# [((1, 0), (2, 0)), ((1, 0), (3, 0)), ((2, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (1, 0)), ((3, 0), (2, 0))]
# ```
# 
# Refer:  http://spark.apache.org/docs/latest/api/python/pyspark.html?highlight=cartesian#pyspark.RDD.cartesian

# #### Definition

# In[13]:


## Insert your answer in this cell. DO NOT CHANGE THE NAME OF THE FUNCTION.
def get_cartesian(rdd):
    ###
    ### YOUR CODE HERE
    ###[(1, 1), (1, 2), (2, 1), (2, 2)]
    return rdd.cartesian(rdd).filter(lambda x : non_duplicates(x))


# #### Unit Tests

# In[14]:


test_rdd = sc.parallelize([(1,0), (2,0), (3,0)])

l = [((1, 0), (2, 0)), ((1, 0), (3, 0)), ((2, 0), (1, 0)), ((2, 0), (3, 0)), ((3, 0), (1, 0)), ((3, 0), (2, 0))]

assert isinstance(get_cartesian(test_rdd), RDD) == True, "Incorrect Return type: Function should return an RDD"
assert set(get_cartesian(test_rdd).collect()) == set(l), "Incorrect Return Value: Value obtained does not match"


# In[15]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[16]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Exercise 4: find_slope

# #### Example
# 
# The function <font color="blue">find_slope</font> computes slope between points A and B and returns it in the format specified below.
# 
# Input: Pair (A,B) where A and B are of form (Ax, Ay) and (Bx, By) respectively. 
# 
# **<font color="magenta" size=2>Example Code</font>**
# ``` python
# my_input = ((1,2),(3,4))
# find_slope(my_input)
# ```
# 
# Output: Pair ((A,slope), B) where A and B have the same definition as input and slope refers to the slope of the line segment connecting point A and B.
# 
# **<font color="blue" size=2>Example Output</font>**
# ``` python
# (((1, 2), 1.0), (3, 4))
# ```
# <font color="brown">**Note: **</font> If Ax == Bx, use slope as "inf".
# 
# <font color="red">**Hint : **</font> The above example is given just to provide the input and output format. This function is called a different way in the spark exercise.
# 

# #### Definition

# In[17]:


## Insert your answer in this cell

def find_slope(x):
    ###
    ### YOUR CODE HERE
    ###
    denom=(x[1][0]-x[0][0])
    slope=  "inf" if (denom==0) else (x[1][1]-x[0][1])/denom
    return ((x[0],slope),x[1])


# #### Unit Tests

# In[18]:


assert type(find_slope(((1,2),(3,4)))) == tuple, "Function must return a tuple"


# In[19]:


assert find_slope(((1,2),(-7,-2)))[0][1] == 0.5, "Slope value should be 0.5"


# In[20]:


assert find_slope(((1,2),(3,4))) == (((1,2),1),(3,4)), "Incorrect return value: Value obtained does not match"


# In[21]:


assert find_slope(((1,2),(1,5))) == (((1,2),"inf"),(1,5)), "Incorrect return value: Value obtained must have slope 'inf'"


# In[22]:


assert find_slope(((1,2),(2,5))) == (((1,2),3),(2,5)), "Incorrect return value: Value obtained does not match"


# In[23]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[24]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[25]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Exercise 5: find_collinear

# #### Example
# 
# The function <font color="blue">find_collinear</font> finds the set of collinear points.
# 
# Input: An RDD (which is the output of the get_cartesian() function. 
# 
# Output: An RDD containing the list of collinear points formatted according to the <font color="blue">format_result</font> function.
# 
# Approach:
# 1. Find the slope of the line between all pairs of points A = (Ax, Ay) and B = (Bx, By).
# 2. For each (A, B), find all points C = ((C1x, C1y), (C2x, C2y), ... (Cnx, Cny)) 
#    where slope of (A,B) = slope of (A, Ci).
# 3. Return (A, B, Ck) where Ck = all points of C which satisfy the condition 1.
# 
# The assert statement unit tests for this function will help you with this.
# <font color="red">**Hint : **</font>   You should use the above helper functions in conjunction with Spark RDD API (refer http://spark.apache.org/docs/latest/api/python/pyspark.html?highlight=rdd#pyspark.RDD)
#             Finally, use helper function format_result() appropriately from inside this function after you have implemented the above operations.

# #### Definition

# In[127]:


def find_collinear(rdd):
    ###
    ### YOUR CODE HERE
    ###
    group = rdd.map(lambda x: find_slope(x))             .groupByKey()             .mapValues(list)             .map(lambda x:tuple(sorted(format_result(x))))             .distinct()             .filter(lambda x: len(x)>2)
    return group


# #### Unit Tests

# In[129]:


def verify_collinear_sets(collinearpointsRDD, testlist):
    collinearpoints = [tuple(sorted(x)) for x in list(set(collinearpointsRDD.collect()))]
    testlist = [tuple(sorted(x)) for x in list(set(testlist))]
    return set(collinearpoints) == set(testlist)


# In[130]:


test_rdd = sc.parallelize([((4, 2), (2, 1)), ((4, 2), (-3, 4)), ((4, 2), (6, 3)), ((2, 1), (4, 2)), ((2, 1), (-3, 4)), ((2, 1), (6, 3)), ((-3, 4), (4, 2)), ((-3, 4), (2, 1)), ((-3, 4), (6, 3)), ((6, 3), (4, 2)), ((6, 3), (2, 1)), ((6, 3), (-3, 4))])
assert isinstance(find_collinear(test_rdd), RDD) == True, "Incorrect return type: Function must return RDD"


# In[131]:


assert verify_collinear_sets(find_collinear(test_rdd), [((2, 1), (4, 2), (6, 3))]), "Incorrect return value: Value obtained does not match"


# In[132]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# #### Unit Tests II : Using the output of get_cartesian(rdd)

# In[133]:


test_rdd = sc.parallelize([(4, -2), (2, -1), (-3,4), (6,3), (-9,4), (6, -3), (8,-4), (6,9)])
test_rdd = get_cartesian(test_rdd)
assert verify_collinear_sets(find_collinear(test_rdd), [((6, -3), (6, 3), (6, 9)), ((2, -1), (4, -2), (6, -3), (8, -4))]), "Incorrect return value: You have not implemented the find_collinear function in Python"


# In[134]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Exercise 6: The build_collinear_set function

# #### Example
# Using the above functions that you have written along with pyspark functions, write the **build_collinear_set** function and returns an RDD containing the set of collinear points.
# 
# Input: RDD containing the given set of points
# 
# Output: RDD containing the set of collinear points
# 
# <font color="red">**Hint : **</font> Remember that the input RDD consists of a set of strings. Remember to pre-process them using the to_tuple function before performing other operations.

# #### Definition

# In[149]:


def build_collinear_set(rdd):
    
    ###
    ### YOUR CODE HERE
    ###
    prerdd=rdd.map(lambda x:to_tuple(x))
    prerdd=get_cartesian(prerdd)
    
    rdd=find_collinear(prerdd)
    
    # Sorting each of your returned sets of collinear points. This is for grading purposes. 
    # YOU MUST NOT CHANGE THIS.
    rdd = rdd.map(to_sorted_points)
    
    return rdd


# #### Unit Tests

# In[151]:


test_rdd = sc.parallelize(['4 -2', '2 -1', '-3 4', '6 3', '-9 4', '6 -3', '8 -4', '6 9'])
assert isinstance(build_collinear_set(test_rdd), RDD) == True, "build_collinear_set should return an RDD."


# ### The process function

# #### Definition

# In[152]:


def process(filename):
    """
    This is the process function used for finding collinear points using inputs from different files
    Input: Name of the test file
    Output: Set of collinear points
    """
    # Load the data file into an RDD
    rdd = sc.textFile(filename)
    
    rdd = build_collinear_set(rdd)
    
    # Collecting the collinear points RDD in a set to remove duplicate sets of collinear points. This is for grading purposes. You may ignore this.
    res = set(rdd.collect())
    
    return res


# #### Unit Tests: Testing the build_collinear_set function using the process function
# NOTE: You may assume that input files do not have duplicate points.

# In[153]:


assert process("../resource/asnlib/public/data.txt") == {((-2, -2), (1, 1), (2, 2), (3, 3)), ((0, 1), (3, 4), (5, 6)), ((0, -3), (0, 1), (0, 5))}, "Your implementation of build_collinear_set is not correct."


# In[154]:


assert process("../resource/asnlib/public/data50.txt") == {((3, 6), (7, 4), (9, 3)), ((1, 6), (3, 6), (4, 6), (7, 6)), 
                                 ((0, 2), (3, 1), (6, 0)), ((1, 0), (2, 0), (5, 0), (6, 0)), 
                                 ((1, 3), (3, 6), (5, 9)), ((0, 8), (4, 6), (6, 5)), 
                                 ((6, 0), (6, 1), (6, 5), (6, 9)), 
                                 ((7, 2), (7, 3), (7, 4), (7, 6), (7, 8)), ((3, 1), (3, 3), (3, 6)), 
                                 ((0, 2), (1, 2), (5, 2), (7, 2)), ((0, 3), (2, 5), (3, 6), (6, 9)), 
                                 ((0, 2), (1, 3), (2, 4), (4, 6), (5, 7)), ((1, 2), (4, 3), (7, 4)), 
                                 ((0, 3), (4, 6), (8, 9)), ((9, 3), (9, 4), (9, 5)), ((2, 5), (5, 7), (8, 9)), 
                                 ((0, 5), (2, 4), (4, 3), (8, 1)), ((0, 8), (1, 6), (2, 4)), 
                                 ((3, 6), (5, 2), (6, 0)), ((5, 9), (6, 9), (8, 9)), 
                                 ((0, 8), (1, 8), (7, 8)), ((0, 4), (1, 3), (3, 1)), ((5, 9), (7, 6), (9, 3)), 
                                 ((1, 2), (2, 4), (3, 6)), ((0, 7), (1, 5), (3, 1)), 
                                 ((1, 5), (2, 4), (3, 3), (6, 0)), ((0, 2), (3, 3), (9, 5)), 
                                 ((0, 7), (1, 6), (2, 5), (4, 3), (5, 2), (6, 1)), 
                                 ((0, 4), (1, 5), (5, 9)), ((1, 5), (3, 6), (5, 7), (7, 8)), 
                                 ((1, 6), (3, 3), (5, 0)), ((3, 6), (4, 3), (5, 0)), 
                                 ((1, 2), (4, 5), (7, 8), (8, 9)), ((0, 2), (1, 1), (2, 0)), 
                                 ((3, 3), (4, 5), (5, 7), (6, 9)), ((0, 2), (0, 3), (0, 4), (0, 5), (0, 7), (0, 8)), 
                                 ((2, 0), (4, 3), (8, 9)), ((5, 7), (6, 5), (7, 3), (8, 1)), ((5, 0), (7, 6), (8, 9)), 
                                 ((5, 0), (6, 1), (7, 2), (9, 4)), ((0, 4), (1, 2), (2, 0)), 
                                 ((1, 1), (3, 1), (6, 1), (8, 1)), ((5, 7), (7, 6), (9, 5)), ((1, 1), (7, 4), (9, 5)), 
                                 ((0, 4), (2, 4), (7, 4), (9, 4)), ((1, 0), (3, 1), (5, 2), (7, 3), (9, 4)), 
                                 ((2, 0), (3, 3), (4, 6), (5, 9)), ((4, 3), (4, 5), (4, 6)), 
                                 ((1, 0), (4, 3), (6, 5), (7, 6)), ((0, 3), (2, 4), (4, 5)), 
                                 ((1, 6), (4, 5), (7, 4)), ((1, 0), (1, 1), (1, 2), (1, 3), (1, 5), (1, 6), (1, 8)), 
                                 ((0, 3), (1, 3), (3, 3), (4, 3), (7, 3), (9, 3)), ((0, 4), (2, 5), (4, 6)), 
                                 ((0, 7), (3, 6), (6, 5), (9, 4)), ((1, 8), (4, 6), (7, 4)), 
                                 ((0, 5), (3, 3), (6, 1)), ((1, 8), (3, 6), (4, 5), (7, 2), (8, 1)), 
                                 ((1, 2), (3, 1), (5, 0)), ((1, 1), (5, 2), (9, 3)), 
                                 ((5, 0), (5, 2), (5, 7), (5, 9)), ((0, 5), (1, 5), (2, 5), (4, 5), (6, 5), (9, 5)), 
                                 ((3, 1), (4, 5), (5, 9)), ((2, 0), (2, 4), (2, 5)), ((5, 2), (6, 5), (7, 8))}, "Your implementation of build_collinear_set is not correct."


# In[155]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[156]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[157]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[158]:


##Hidden test cases here
###
### AUTOGRADER TEST - DO NOT REMOVE
###

