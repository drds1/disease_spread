# disease_spread


This project models the UK COVID19 daily case rate I(t) as an exponential process of the form

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0e^{m(t-t_0)}" title="equation_I}" />

with the reference daily case rate I_0 and growth constant m as parameteres to be modelled.

To ensure any modelling reflects the most recent status of the model parameters, the above equation should be fitted to a relatively recent timeseries of daily case rates. Too short a fitting period and we will get poor inferences of the model parameters with large uncertainties. Too long and we risk confusing the fit by including data from earlier in the pandemic where the growth rate behaved differently (corresponding to a different set of I_0 and m parameters). For this model, we fit the most recent 21 days of case rates.

Linear models have the advantage of an analytically-derived fit without requiring iteration and we can cast the above relation as a linear model by taking natural logs of both sides such that, 

<img src="https://latex.codecogs.com/svg.latex?\Large&space;ln(I)=ln(I_0) + m(t-t_0)" title="equation_lnI}" />

Covariance between the ln(I_0) and m parameters can then be eliminated by chosing a suitable choice of a suitable feature space such that
<img src="https://latex.codecogs.com/svg.latex?\Large&space;t-t_0 = -N/2 ... N/2" title="equation_t0}" />

where N is the modelled sample size (21 days) and we increment in 1-day cadence (e.g. t - t_0 = -10, -9, ... 8, 9, 10).



## Derivation of R

The above is equivalent to the following relation involving the popular reproduction factor R formulation of the exponential growth problem where

<img src="https://latex.codecogs.com/svg.latex?\Large&space;I(t)=I_0R^{t/\tau}" title="equation_r" />

R can then be related to the inf


![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/forecast.png)


## Daily UK Case Data by Specimin Date
The daily case data, available at https://coronavirus.data.gov.uk/ , is refreshed daily at around 16:30 GMT and can be used to produce daily updates to the following forecast models to ensure both the forecast and R parameter inferences are as recent as possible.


## Inter





## Parameter fits

The best fitting parameters and uncertainties are plotted below. We see that decorrelating the parameters using a suitably chosen  pivot point t_0 has eliminated almost all the covariance between the R and I_0 parameters. The large uncertainty in R arrises mainly due to few (21) and highly scattered data points in our modelled sample.

![Test Image 1](https://github.com/dstarkey23/disease_spread/blob/master/results/recent/correlation.png)

