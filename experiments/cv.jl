using SubsetSelection, DataFrames, LinearAlgebra

function split_data(n, holdout)
    batch = ceil(Int, n*holdout)
    perm = shuffle(collect(1:n))
    train = perm[1:(n-batch)]
    val = perm[(n-batch+1): n]
    return train, val
end

function error(ℓ::Regression, Y, X, indices, w)
  Z = Y .- X[:,indices]*w
  return dot(Z,Z)/size(Z,1)
end

function error(ℓ::Classification, Y, X, indices, w)
    df = DataFrame(Output = (Y.>0), Probability = 1. ./ (1 .+ exp.(X[:,indices]*w)))
    sort!(df, :Probability)
    # sort!(df, cols = :Probability)
    # df = sort(df, cols = :Probability)
    df[!,:Above]=(1:length(Y)) .- cumsum(df[:,:Output])

    nP = length(findall(o->o>0, Y)); nN = length(findall(o->o<0, Y));

    AUC = 1-sum(df[:,:Above].*df[:,:Output])/nP/nN
    return 1-AUC
end

function error2(ℓ::Classification, Y, X, indices, w)
    Y_truth = Y .> 0
    Y_pred = X[:,indices]*w .> 0

    MR = sum(1 .- Y_truth.*Y_pred .- (1 .- Y_truth).*(1 .- Y_pred)) / size(Y,1)
    return MR
end

function grad_primal(ℓ::LossFunction, Y, X, w, γ)
  n,k = size(X)
  g = zeros(k)
  for j in 1:k
    g[j] += w[j] / γ
  end
  for i in 1:n
    g += grad_loss(ℓ, Y[i], X[i,:], w)
  end
  return g
end

function grad_loss(ℓ::OLS, y, x, w)
  return -(y-dot(x,w))*x
end
function grad_loss(ℓ::L1SVM, y, x, w)
  return -y*(1-y*dot(x,w) > 0)*x
end
function grad_loss(ℓ::LogReg, y, x, w)
  z = exp.(y*dot(x,w))
  return -y/(1+z)*x
end

function value_primal(ℓ::LossFunction, Y, X, w, γ)
  n,k = size(X)
  v = dot(w,w)/2/γ
  for i in 1:n
    v += value_loss(ℓ, Y[i], X[i,:], w)
  end
  return v
end

function value_loss(ℓ::OLS, y, x, w)
  return .5*(y-dot(x,w))^2
end
function value_loss(ℓ::L1SVM, y, x, w)
  return max(0, 1-y*dot(x,w))
end
function value_loss(ℓ::LogReg, y, x, w)
  z = exp.(-y*dot(x,w))
  return log(1+z)
end
