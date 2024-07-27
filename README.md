# Titanic Survivability Prediction

This project was developed by:
* Afonso Coelho ([Bugss05](https://github.com/Bugss05)) - FCUP_IACD:202305085
* Diogo Amaral ([damaral31](https://github.com/damaral31)) - FCUP_IACD:202305187
* Miguel Carvalho ([miguel-c05](https://github.com/miguel-c05)) - FCUP_IACD:202305229

## Context and Objective
This repository serves as the workspace for the Kaggle competition ["Titanic - Machine Learning from Disaster"](https://www.kaggle.com/competitions/titanic), a Machine Learning exercise best suited for beginners, especially those new to the Kaggle platform.

The aim of the competition at hand is to study a Database of passengers aboard the Titanic and train a Machine Learning model on any relevant information extracted in order to be able to predict whether a passenger is likely to survive the ship's wreck or not. All of this with the maximum accuracy possible, of course.

## The Data
In order to train the model a large enough amount of information is needed. As such, Kaggle gives the participants of this competition a dataset ([train.csv]()) on which to work, regarding passenger's names, age, sex and nº of siblings, among others. It also contains, however, missing values as well as outliers, both of which harm data analysis and model training. Later, we will explain how such cases were handled.

## Expectations
On a first inspection of the facts, women and children will most likely have a higher chance of survival, simply because they were groups of people prioritized for evacuation and rescue. Some cabins may also see a survivability increase solely due to its geographical location on the ship.

## Missing Values
To effectively handle missing values, multiple methods were researched according top the well-known paper ["Improved Heterogeneous Distance Functions", D. Randall Wilson, Tony R. Martinez](https://www.jair.org/index.php/jair/article/view/10182/24168). As such, Heterogeneous  Value  Difference  Metric (HVDM) was chosen, not only because of its efficiency, but also due to its easy implementation. This algorithm can be summarized by the following expressions:

$$
HVDM(x, y) = \sqrt{\sum_{a=1}^{m} d_a^2 (x_a, y_a)}
$$

<br>

$$
d_a(x, y) = 
\begin{cases} 
1, & \text{if } x \text{ or } y \text{ is unknown}; \text{ otherwise...} \\ 
normalizedVdm_a(x, y), & \text{if } a \text{ is nominal} \\ 
normalizedDiff_a(x, y), & \text{if } a \text{ is linear}
\end{cases}
$$

<br>

$$
normalizedDiff_a(x, y) = \frac{|x - y|}{4\sigma_a}
$$

<br>

$$
normalizedVdm_{a}(x,y) = \sqrt{\sum_{c=1}^{C} \left| \frac{N_{a,x,c}}{N_{a,x}} - \frac{N_{a,y,c}}{N_{a,y}} \right|^2}
$$


<br>

where:
* $x$ and $y$ are passengers;
* $m$ is the nº of attributes of a passenger;
* $a$ is an attribute;
* $\sigma$ is the standard deviation
* $c$ is the output class
* $C$ is the nº of output classes

<br>

So, in laymen's terms, HVDM finds the distance between two passengers by comparing the difference of each attribute and adding them. It is similar to the HEOM algorithm, differing mainly in that HVDM normalizes its values before comparing them.
<br><br>
Finally, having the distances between all passengers, a missing value in a said attribute will be filled by searching the "closest" $K$ (where $K \in \mathbb{N}$) passengers and calculating their average, similarly to the KNN algorithm.

## Outliers
bla bla bla bla bla bla bla bla bla bla bla bla


## Machine Leaning Algorithm
bla bla bla bla bla bla bla bla bla bla bla bla

### Alg. 1
bla bla bla bla bla bla bla bla bla bla bla

### Alg. 2
bla bla bla bla bla bla bla bla bla bla bla

### Alg. 3
bla bla bla bla bla bla bla bla bla bla bla

### Ensembling
bla bla bla bla bla bla bla bla bla bla bla


## Installation and Usage
bla bla bla bla bla bla bla bla bla bla bla


## References
[Wilson, D. R., & Martinez, T. (1997). Improved heterogeneous distance functions. Journal of Artificial Intelligence Research, 6(1) 1-34.](https://jair.org/index.php/jair/article/view/10182)
