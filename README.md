# SubsetSelectionCIO

SubsetSelection is a Julia package that computes sparse L2-regularized estimators. Sparsity is enforced through explicit cardinality constraint. Supported loss functions for regression are least squares; for classification, logistic and L1 Hinge loss. The algorithm formulates the problem as a pure-integer convex optimization problem and solves it using a cutting plane algorithm.The package is built up on the [SubsetSelection](https://github.com/jeanpauphilet/SubsetSelection.jl) package. 

## Quick start
To install the package:
```julia
julia> Pkg.clone("git://github.com/jeanpauphilet/SubsetSelectionCIO.jl.git")
```
To fit a basic model:

```julia
julia> using SubsetSelectionCIO, StatsBase

julia> n = 100; p = 10000; k = 10;
julia> indices = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
julia> w = sample(-1:2:1, k);
julia> X = randn(n,p); Y = X[:,indices]*w;
julia> γ = 1/sqrt(size(X,1));
julia> indices0, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(), Y, X, k, γ)
([36, 184, 222, 240, 325, 347, 361, 605, 957, 973], [-0.950513, -0.94923, -0.950688, -0.956536, 0.951954, -0.953707, -0.954927, -0.9571, -0.959357, -0.95312], 0.26711583137512207, :Optimal, 0.0, 17)
```

The algorithm returns a set of indices `indices0`, the value of the estimator on the selected features only  `w0`, the time needed to compute the model `Δt`, the status of the MIP solver `status`, the sub-optimality gap `Gap` and the number of cuts required by the cutting-plane algorithm `cutCount`.

For classification, we use +1/-1 labels and the convention 
`P ( Y = y | X = x ) = 1 / (1+e^{- y x^T w})`.

## Required and optional parameters

`oa_formulation` has five required parameters:
- the loss function to be minimized, to be chosen among least squares (`SubsetSelection.OLS()`), Logistic loss (`SubsetSelection.LogReg()`) and Hinge Loss (`SubsetSelection.L1SVM()`). 
- the vector of outputs `Y` of size `n`, the sample size. In classification settings, `Y` should be a vector of ±1s.
- the matrix of covariates `X` of size `n`×`p`, where `n` and `p` are the number of samples and features respectively.
- the level sparsity `k`; the algorithm will consider the hard constraint "||w||_0 < k".
- the value of the ℓ2-regularization parameter `γ`.

In addition, `oa_formulation` accepts the following optional parameters:
- an initialization for the selected features, `indices0`.
- a time limit `ΔT_max`, in seconds, set to 60 by default.
- `verbose`, a boolean. If true, the MIP solver information is displayed. By default, set to false.
- `Gap` a limit on the suboptimality gap to reach. By default, set to 0.  

## Best practices
- Tuning the regularization parameter `γ`: Setting `γ` to 1/√n seems like an appropriate scaling in most regression instances. For an optimal performance, and especially in classification or noisy settings, we recommend performing a grid search and using cross-validation to assess out-of-sample performance. The grid search should start with a very low value for `γ`, such as  
```julia 
    γ = 1.*p / k / n / maximum(sum(X[train,:].^2,2))
``` 
and iteratively increase it by a factor 2. Mean square error or Area Under the Curve (see [ROCAnalysis]( https://github.com/davidavdav/ROCAnalysis.jl) for implementation) are commonly used performance metrics for regression and classification tasks respectively.
- The mixed-integer solver greatly benefits from a good warm-start, even though the warm start is not feasible, i.e., `k`-sparse. Among other methods, one can use the output of [SubsetSelection](https://github.com/jeanpauphilet/SubsetSelection.jl) or a Lasso estimator (see [GLMNet](https://github.com/JuliaStats/GLMNet.jl) implementation for instance).

## Reference
Dimitris Bertsimas, Jean Pauphilet, Bart Van Parys, <i> Sparse Classification : a scalable discrete optimization perspective <i/>, available on [Arxiv](http://arxiv.org/abs/1710.01352)
