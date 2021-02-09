# COVID Case Forecasts

With COVID-19 still (at the time of writing) very much ravaging the UK, almost all the population remains in tight lockdown. This project aims to forecast an approximate date at which travel restrictions might slowly and safely begin to unwind. 

This project models the UK COVID19 daily case rate I(t) as an exponential process of the form

**1:** <img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0e^{m(t-t_0)}" title="equation_I}" />  

with the reference daily case rate *I<sub>0</sub>* and growth constant m as parameteres to be modelled. It is assumed that once daily cases drop below an arbitrary threshold of 1000 new cases per day, lockdown magically ends.


## Daily Case Data

This is a publically available UK government data source available at https://coronavirus.data.gov.uk/ . An machine-readable, code-integrable api call for the 'case-by-specimen' data can be found at https://api.coronavirus.data.gov.uk/v2/data?areaType=overview&metric=newCasesBySpecimenDate&format=csv .


## Derivation of R

Equation 1 is equivalent to the following relation involving the reproduction factor *R* formulation of the exponential growth problem where

**2:** <img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0R^{(t-t_0)/\tau}" title="equation_I_r" />


In reality, an individual is unlikely to be equally as infectious throughout their illness and will exhibit a time dependent "infectivity" curve. This will be neglected here and a uniform infectious period of &tau; = 14 days is assumed during which an individual is equally likely to pass along the disease. This in effect assumes a top-hat (or boxcar) infectivity curve. 

*R*, by equating Equations 1 and 2, is given by *R=e<sup>m/&tau;</sup>*. 


## Model Data Selection

To ensure our forecasts are based on current pandemic behaviour, Equation 1 should be fitted to only the most recent segment of data from the case rate timeseries. Including too few dates in our model will yield inprecise parameter inferences whereas too long and we risk confusing the fit by including data from earlier in the pandemic with different growth behaviour (corresponding to a different set of I_0 and m parameters). In this model we fit the most recent 21 days of case rates.


## Log-linear Modelling

Linear models have the advantage of an analytically-derived fit without requiring iteration and we can cast the above relation as a linear model by taking natural logs of both sides such that, 

**3:** <img src="https://latex.codecogs.com/svg.latex?\Large&space;ln(I)=ln(I_0)+m(t-t_0)" title="equation_lnI}" />

Covariance between the *ln(I<sub>0</sub>)* and *m* parameters can then be eliminated by a suitable choice of feature space such that
*t-t<sub>0</sub> = -N/2 ... N/2*, where *N* is the modelled sample size (21 days) and we increment in 1-day cadence (e.g. *t - t<sub>0</sub> = -10, -9, ... 8, 9, 10*).



![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/forecast.png)


*numpy*'s polyfit module is used to perform the log linear fitting. This is embedded into the *rmodel* class in the [accompanying script](https://github.com/dstarkey23/disease_spread/blob/master/rvalue_model.py).



## Model Uncertainties

The model also attempts to simulate the expected uncertainty in the forecast using the day-to-day scatter between case rates. This is achieved by assigning inverse variance weights to each data point whose variances are calculated from the residuals of a smooth 5-day rolling average.





## Parameter fits

The best fitting parameters and uncertainties are plotted below. We see that decorrelating the parameters using a suitably chosen  pivot point t_0 has eliminated almost all the covariance between the *R* and *I<sub>0</sub>* parameters. The large uncertainty in *R* arrises mainly due to few (21) and highly scattered data points in our modelled sample.

![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/correlation.png)






## Appendix

Notes for embedding equations and symbold in .md files: https://www.xspdf.com/help/50653644.html
e.g. *I(t)=I<sub>0</sub> R<sup>(t-t<sub>0</sub>)/&tau;</sup>*

