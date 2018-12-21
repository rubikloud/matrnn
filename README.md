# Multivariate Arrival Times Recurrent Neural Network

Current approaches to modeling arrival times 
	are ad-hoc implementations
	of neural networks or tree-based predictors.
These are not particularly built for the problem
	of dealing with partially observed data.
In survival analysis, for example,
	a majority of the observations are unobserved
	(i.e. subjects have not died yet)
	but we want to estimate their age-of-death
	in the interest of prolonging it.

Similar to `WTTE-RNN`
	([linked here](https://github.com/ragulpr/wtte-rnn/)),
	we adopt a parametric approach.
However, we do not attempt to model the hazard rate,
	hence we do not require strict assumptions 
	such as memoryless-ness,
	which is a problem encountered in `WTTE-RNN`.
We assume that there are only as many arrival times as there are events
	and we observe conditional versions of them
	for each time `t`.


## Conditional Excess Random Variable

First we define the conditional excess random variable `Z_t`.
This is the remaining time-to-event
	conditioned on the true time-between-event
	exceeding that which has already been observed
	(i.e. the time since previous event).
Let `Y_i` be the time between the `i-1`-th and `i`-th arrival
	and `N(t)` be the number of arrivals by time `t`.
We are interested in `Y_{N(t)+1}`,
	which is the time to the next arrival.
Hence we define the conditional excess random variable as follows.


`Z_t = Y_{N(t)+1} - time-since-event(t) | Y_{N(t)+1} > time-since-event(t)`



## Loss Function

Since the recurrent neural network (RNN) 
	outputs parameters `k_t` at each time `t`
	for the distribution of the true time to next arrival `Y_{N(t)+1}`,
	we can define a likelihood loss,
	which is the quality of prediction at time `t`.
Suppose that `f(.|k), S(.|k)`
	are the density `f(t) = lim_{t to 0} P(t-h < Y < t+h) / 2h` 
	and survival functions `S(t) = P(Y > t)`
	for `Y` which are parametrized by `k`.
Then this induces a distribution on `Z_t = Y - tse(t) | Y > tse(t)`
	which we assume are `f*(.|k), S*(.|k)`.


If the next time-to-event is observed before the end of training period,
	we have an uncensored observation.


`loss_t = f*(time-to-event(t)|k_t)`


Otherwise, we have a censored observation.


`loss_t = S*(time-to-end(t)|k_t)`


## Implementation

In 
	[`examples/CMAPSS/`](https://github.com/tianle91/matrnn/tree/master/examples/CMAPSS/)
	you'll find code to run the model as described
	with `tensorflow` and `keras`.

- `matrnn_distributional.py` has functions for the Weibull distribution to use outside of training.
- `matrnn_objective.py` describes the log-likelihoods used in model training.
- `matrnn_fitter.py` defines the methods for training and inference.

For the input matrix, the loss function expects a `y` matrix of shape
	`(n_observations, n_sequence, n_eventtypes, 4)`.
The last axis contains all the arrival-time information for the process 
	you want to model
	`4 = length((tse, tte, unc, purchstatus))`.

- `tse(t)` is time since previous event if `purchstatus(t)==1` and time since start if otherwise
- `tte(t)` is time to next event if `unc(t)==1` and time to end of training if otherwise
- `unc(t)` is `1` if next event is observed by end of training and `0` otherwise
- `purchstatus(t)` is `1` if first event has occurred and `0` otherwise

You'll need to tweak the fitting and training methods for production.
The generation of the `(tse, tte, unc, purchstatus)` time-series
	converts a sparse table of arrivals into a dense matrix,
	making this an i/o bottleneck.


# What Can You Do/Not-Do With MAT-RNN?

A distribution for the prediction for next arrival time is useful
	when you have sparse and highly dependent multivariate arrival times.


## Do: Any function of next arrival time.

You can set your estimates to minimize some exogenous loss function
	that is based on a random time-to-arrival
	by minimizing the expected loss,
	or output probabilities of arrival in some time period
	as well as a point estimate (take the mode of prediction).


## Not-Do: Any function of multiple next arrival times.

A naive approach would be to bootstrap your estimates
	by consecutively running predicted times to next arrivals
	until the end of the time period
	but this is likely to give massive variances in your estimates.
If you're expecting a few (i.e. more than one) arrivals
	in each time period,
	it might be better to consider a stochastic intensity model.
I'm quite sure there are a few stochastic intensity models
	using RNNs but I can do it if there aren't any.


## Cite Me
```
@inproceedings{Chen2018MultivariateForecasting,
    title = {{Multivariate Arrival Times with Recurrent Neural Networks for Personalized Demand Forecasting}},
    year = {2018},
    booktitle = {Proceedings of International Conference of Data Mining DMS Workshop},
    author = {Chen, Tianle and Keng, Brian and Moreno, Javier}
}
```