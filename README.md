Python code for some covariance estimators.

* LWO_estimator: Linear shrinkage estimator following [Ledoit and Wolf, 2004] when the mean is known, and [Oriol and Miot, 2025] when it is unknown.
* MTSE:	Multi-target linear shrinkage estimator, following [Oriol, 2023].
* analytical_shrinkage: Analytical non-linear shrinkage estimator using kernel density estimation, following [Ledoit and Wolf, 2020] when samples are uniformly weighted.
* QIS: Quadratic-Inverse Shrinkage estimator, following [Ledoit and Wolf, 2020], code inspired from Patrick Ledoit's implementation (github.com/pald22/covShrinkage)
* LIS: Linear-Inverse Shrinkage estimator, following [Ledoit and Wolf, 2020], code inspired from Patrick Ledoit's implementation (github.com/pald22/covShrinkage)
* GIS: Geometric-Inverse Shrinkage estimator, following [Ledoit and Wolf, 2020], code inspired from Patrick Ledoit's implementation (github.com/pald22/covShrinkage)
* NERCOME: NERCOME estimator from [Lam, 2016]
* tyler_estimation: Tyler M-estimator for scatter matrix [Tyler et al., 1987] and variants - SRTy estimator [Breloy et al., 2019], a shrinkage Tyler estimator, and Ashurbekova estimators for elliptic distributions [Ashurbekova et al., 2019] (only Student and Gaussian here) - along with tail estimators to form covariance estimators.

* CCC: Constant Conditional Correlation [Bollerslev, 1990]
* DCC: Dynamic Conditional Correlation [Engle, 2002] originally, [Pakel, 2021] for 2MSCLE, [Engle et al., 2019] for targetted correlation improvement

* QuEST: implementation of the QuEST function following [Ledoit and Wolf, 2017].

Estimators implemented in ScikitLearn are not reproduced here - it includes OAS, Empirical Covariance, Shrunk Covaricance, GLasso, MinCovDet, Elliptic Envelope.
