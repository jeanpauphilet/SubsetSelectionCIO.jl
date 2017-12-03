# SubsetSelectionCIO

SubsetSelection is a Julia package that computes sparse L2-regularized estimators. Sparsity is enforced through explicit cardinality constraint. Supported loss functions for regression are least squares; for classification, logistic and L1 Hinge loss. The algorithm formulates the problem as a pure-integer convex optimization problem and solves it using a cutting plane algorithm.

## Quick start
To install the package:
```julia
julia> Pkg.clone("git://github.com/jeanpauphilet/SubsetSelectionCIO.jl.git")
```
