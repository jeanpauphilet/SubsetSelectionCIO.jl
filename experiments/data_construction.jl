using Random, LinearAlgebra, StatsBase, Distributions

##FUNCTION data_construction
"""Generate synthetic data
INPUTS
- ℓ             Loss function
- n             Number of samples
- k             Number of relevant features
- p             Number of features/covariates
- SNR           Squareroot of the Signal-to-Noise Ratio
- rho           Correlation coefficient between features
OUTPUTS
- X             Matrix of observed covariates
- Y             Vector of responses
- indices_true  Set of relevant features
- w_true        True regressor"""
function data_construction(ℓ::LossFunction, n::Int, k::Int, p::Int, SNR, ρ)
  indices_true, w_true = generate_estimator(p, k)
  #Auto-regresive AR(1) filter preserving noise power
  X = randn(n, p)
  for i in 2:p
      X[:, i] = X[:, i] + ρ*X[:, i-1]
  end
  # rho = ρ
  # R"""
  # X=matrix(NA,$n,$p)
  # for(i in 1:$n){
  #     X[i,]=arima.sim(n=$p,list(ar=$rho))
  # }
  # """
  # @rget X
  # X = √12 * rand(n, p) - (√12)/2

  Y = generate_output(ℓ, noisy_signal(X, indices_true, w_true, SNR))

  return X, Y, indices_true, w_true
end

function hard_data_construction(ℓ::LossFunction, n::Int, k::Int, p::Int, SNR, ρ)
  indices_true, w_true = generate_estimator(p, k)

  indices_true = collect(1:k)
  w_true = ones(k)./√k #sample(-1:2:1, k)

  #Covariance matrix
  θ = (1/√k + 1/k)/2
  Σ = SparseMatrixCSC{Float64}(I, p, p)
  for i in 1:k
    Σ[i,k+1] = θ
    Σ[k+1,i] = θ
  end

  # perm = shuffle(1:p)
  # indices_true = [findfirst( perm .== i) for i in indices_true]
  # Σ[:,:] = Σ[perm, perm]

  d = MvNormal(full(Σ))

  X = rand(d, n)'

  Y = generate_output(ℓ, noisy_signal(X, indices_true, w_true, SNR))

  # E = randn(n, p)  #Noise
  # for j in 1:p
  #   E[:,j] *= norm(X[:,j])/norm(E[:,j])/SNR*2
  # end

  return X, Y, indices_true, w_true
end

# FUNCTIONS Generate support and regressor w
function generate_estimator(p::Int, k::Int)
  weights = ones(p) ./ p
  indices = sort(sample(1:p, StatsBase.Weights(weights), k, replace=false))
  w = sample(-1:2:1, k)
  return indices, w
end

function noisy_signal(X, indices, w, SNR)
  n = size(X,1)
  S = X[:, indices]*w #Signal
  E = randn(n)  #Noise
  E *= (norm(S)/norm(E))/SNR #Scaling according to SNR
  return S+E
end

#FUNCTIONS Generate Output Y
function generate_output(ℓ::Regression, S)
  return S
end

function generate_output(ℓ::L1SVM, S)
  P = 1. ./( 1 .+ exp.(-S) )
  return 1.0*(P .> .5) - 1.0*(P .< .5)
end

function generate_output(ℓ::LogReg, S)
    n = size(S,1)
    P = 1. ./( 1 .+ exp.(-S) )
    return (-ones(n)).^(rand(n) .> P)
end
