using SubsetSelection
using SubsetSelectionCIO
using Test
using StatsBase

n = 500; p = 1000; k = 10;
indices = sort(sample(1:p, StatsBase.Weights(ones(p)/p), k, replace=false));
w = sample(-1:2:1, k);
X = randn(n,p); Y = X[:,indices]*w;

γ = 1/sqrt(size(X,1))

indices0, w0, Δt, status, Gap, cutCount = oa_formulation(SubsetSelection.OLS(), Y, X, k, γ)
for i in 1:k
  @test indices0[i]==indices[i]
end
