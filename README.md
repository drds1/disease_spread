# disease_spread


This project models the UK COVID19 daily case rate I(t) as an exponential process of the form

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0e^{m(t-t_0)}" title="equation_I}" />  

with the reference daily case rate I_0 and growth constant m as parameteres to be modelled.


## Derivation of R

The above is equivalent to the following relation involving the popular reproduction factor R formulation of the exponential growth problem where

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0R^{(t-t_0)/\tau}" title="equation_I_r" />

In reality, an individual is unlikely to be equally as infectious throughout their illness and will exhibit a time dependent "infectivity" curve. This will be neglected here and a uniform infectious period of \tau = 14 days is assumed during which an individual is equally likely to pass along the disease. This in effect assumes a top-hat (or boxcar) infectivity curve.

 R is obtained from Equations 1 and 2 as 
<img src="https://latex.codecogs.com/svg.latex?\Large&space;R=e^{m\tau}" title="equation_r" />



## Model Data Selection

To ensure our forecasts are based on current pandemic behaviour, Equation 1 should be fitted to only the most recent segment tof the case rate timeseries. Including too few dates in our model will yield poor inferences of the model parameters with unhelpfully large uncertainties. Too long and we risk confusing the fit by including data from earlier in the pandemic with different growth behaviour (corresponding to a different set of I_0 and m parameters). In this model we fit the most recent 21 days of case rates.


## Log-linear Modelling

Linear models have the advantage of an analytically-derived fit without requiring iteration and we can cast the above relation as a linear model by taking natural logs of both sides such that, 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;ln(I)=ln(I_0)+m(t-t_0)" title="equation_lnI}" />

Covariance between the ln(I_0) and m parameters can then be eliminated by chosing a suitable choice of a suitable feature space such that
<img src="https://latex.codecogs.com/svg.latex?\Large&space;t-t_0 = -N/2 ... N/2" title="equation_t0}" />

where N is the modelled sample size (21 days) and we increment in 1-day cadence (e.g. t - t_0 = -10, -9, ... 8, 9, 10).



![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/forecast.png)


## Daily UK Case Data by Specimin Date
The daily case data, available at https://coronavirus.data.gov.uk/ , is refreshed daily at around 16:30 GMT and can be used to produce daily updates to the following forecast models to ensure both the forecast and R parameter inferences are as recent as possible.


## Inter





## Parameter fits

The best fitting parameters and uncertainties are plotted below. We see that decorrelating the parameters using a suitably chosen  pivot point t_0 has eliminated almost all the covariance between the R and I_0 parameters. The large uncertainty in R arrises mainly due to few (21) and highly scattered data points in our modelled sample.

![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/correlation.png)

