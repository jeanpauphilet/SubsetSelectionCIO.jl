using SubsetSelection

####################
## OLS
####################

# FUNCTION inner_op
# """Computes the exact regression error for a given selection s of features.
#
# INPUT
#   ℓ           - LossFunction to use
#   Y           - Vector of outputs. For classification, use ±1 labels
#   Z           - Array of inputs
#   s           - Sparsity pattern. 0/1 vector
#   γ           - Regularization paramter
#
# OUTPUT
#   α           - Dual variables
#   c           - function value c(s)
#   ∇c          - subgradient of c at s"""
# function inner_op(ℓ::OLS, Y, X, s, γ)
#
#   indices = find(s->s>0.5, s); k = length(indices)
#   n,p = size(X)
#
#   # Compute optimal dual parameter
#   α = sparse_inverse(ℓ, Y, X[:, indices], γ)
#
#   c = SubsetSelection.dual(ℓ, Y, X, α, indices, k, γ)
#
#   ∇c = zeros(p)
#   for j in 1:p
#     ∇c[j] = -γ/2*dot(X[:,j],α)^2
#   end
#
#   return c, ∇c
# end


# FUNCTION sparse_inverse
"""Computes the optimal dual variables
          α = (∑ γ X_j X_j' + I)^(-1) Y
using the Matrix Inversion Lemma.

INPUT
  ℓ           - LossFunction to use
  Y           - Vector of observed responses
  X           - Matrix of observed features
  γ           - Regularization parameter

OUTPUT
  α           -Optimal dual variables"""
function sparse_inverse(ℓ::LossFunction, Y, X, γ)

  n = size(X, 1)
  k = size(X, 2)

  CM = eye(k)/γ + X'*X      # The capacitance matrix
  α = -Y + X*(CM\(X'*Y))       # Matrix Inversion Lemma

  return α
end

# # FUNCTION recover_primal
# """Computes the Ridge regressor
#
# INPUT
#   ℓ           - LossFunction to use
#   Y           - Vector of observed responses
#   Z           - Matrix of observed features
#   γ           - Regularization parameter
#
# OUTPUT
#   w           - Optimal regressor"""
# function recover_primal(ℓ::OLS, Y, Z, γ)
#   α = sparse_inverse(ℓ, Y, Z, γ)           # Optimal dual variable α
#   return -γ*Z'*α                            # Regressor
# end

####################
## CLASSIFICATION
####################

# FUNCTION inner_op
"""Computes the exact regression error for a given selection s of features.

INPUT
  ℓ           - LossFunction to use
  Y           - Vector of outputs. For classification, use ±1 labels
  X           - Array of inputs
  s           - Sparsity pattern. 0/1 vector
  γ           - Regularization paramter

OUTPUT
  α           - Dual variables
  c           - function value c(s)
  ∇c          - subgradient of c at s"""
function inner_op(ℓ::LossFunction, Y, X, s, γ)
  indices = find(x->x>0.5, s); k = length(indices)
  n,p = size(X)

  # Compute optimal dual parameter
  α = sparse_inverse(ℓ, Y, X[:, indices], γ)

  c = SubsetSelection.dual(ℓ, Y, X, α, indices, k, γ)

  ∇c = zeros(p)
  for j in 1:p
    ∇c[j] = -γ/2*dot(X[:,j],α)^2
  end

  return c, ∇c
end

using LIBLINEAR

# FUNCTION sparse_inverse
"""Computes the optimal dual variables
          α = (∑ γ X_j X_j' + I)^(-1) Y
using the Matrix Inversion Lemma.

INPUT
  ℓ           - LossFunction to use
  Y           - Vector of observed responses
  X           - Matrix of observed features
  γ           - Regularization parameter

OUTPUT
  α           -Optimal dual variables"""
function sparse_inverse(ℓ::Classification, Y, X, γ;
                                    valueThreshold=1e-8, maxIter=1e3)
    n,k = size(X)
    indices = collect(1:k); n_indices = k
    cache = SubsetSelection.Cache(n, k)

    α = start_primal(ℓ, Y, X, γ)
    value = SubsetSelection.dual(ℓ, Y, X, α, indices, k, γ)
    for iter in 1:maxIter
        ∇ = SubsetSelection.grad_dual(ℓ, Y, X, α, indices, n_indices, γ, cache) #Compute gradient

        learningRate = 2/norm(∇, 1)
        α1 = α
        newValue = value - 1.

        while newValue < value  #Divide step sie by two as long as f decreases
          learningRate /= 2
          α1 = α .+ learningRate*∇       #Compute new alpha
          α1 = SubsetSelection.proj_dual(ℓ, Y, α1)    #Project
          newValue = SubsetSelection.dual(ℓ, Y, X, α1, indices, k, γ)  #Compute new f(alpha, s)
        end

        value_gap = 2*(newValue-value)/(value+newValue)
        α = α1[:]; value = newValue
        if abs(value_gap) < valueThreshold
          break
        end
    end

    return α
end

# FUNCTION start_primal
function start_primal(ℓ::Classification, Y::Array, X::Array, γ::Real)
  n,k = size(X)

  w = SubsetSelection.recover_primal(ℓ, Y, X, γ)
  α = primal2dual(ℓ, Y, X, w)
  return α
end

function primal2dual(ℓ::LogReg, Y, X, w)
  n = size(X, 1)
  return [-Y[i]*exp(-Y[i]*dot(X[i,:], w))/(1+exp(-Y[i]*dot(X[i,:], w))) for i in 1:n] #LR
end

function primal2dual(ℓ::L1SVM, Y, X, w)
  n = size(X, 1)
  return [-Y[i]*(1-Y[i]*dot(X[i,:], w) > 0) - Y[i]/2*(1-Y[i]*dot(X[i,:], w) == 0) for i in 1:n] #SVM
end

# # FUNCTION recover_primal
# """Computes the Ridge regressor
#
# INPUT
#   ℓ           - LossFunction to use
#   Y           - Vector of observed responses
#   Z           - Matrix of observed features
#   γ           - Regularization parameter
#
# OUTPUT
#   w           - Optimal regressor"""
# function recover_primal(ℓ::Classification, Y, Z, γ)
#   solverNumber = LibLinearSolver(ℓ)
#   model = linear_train(Y, Z'; verbose=false, C=γ, solver_type=Cint(solverNumber))
#   return Y[1]*model.w
# end
#
# function LibLinearSolver(ℓ::LogReg)
#   return 7
# end
# function LibLinearSolver(ℓ::L1SVM)
#   return 3
# end
