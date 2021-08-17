using SubsetSelection, SubsetSelectionCIO
using DataFrames
include("cv.jl")

##############################################
##SOLVER FOR CROSS-VALIDATING k/λ
##############################################
function oa_cv_constraint(ℓ::LossFunction, kRange, Γmax, Y, X;
    holdout = .3,
    anticycling = false, averaging = true, gradUp = 10, maxIter = 100, δ = 1e-3,
    TimeLimit = 60.)

  Path = DataFrame(k=Float64[], indices=Array[], w=Array[],
             cuts=Integer[], γ=Float64[], Δt = Float64[], error=Float64[])

  n, p = size(X)

  train, val = split_data(n, holdout) #Split data into train and validation sets

  indInit = SubsetSelection.ind_init(Constraint(kRange[1]), p)
  αInit=SubsetSelection.alpha_init(ℓ, Y[train]) #Initialization

  for c in 1:length(kRange)
    # indInit = partial_min(Constraint(Range[c]), X[train, :], αInit, γ)
    k = kRange[c]
    γ0 = 1. *p / k / (maximum(sum(X[train,:].^2,2))*n)
    factor = 1.

    stop = false
    for inner_epoch in 1:Γmax
      if !stop
        γ = factor * γ0

        result = subsetSelection(ℓ, Constraint(k), Y[train], X[train, :],
            indInit = indInit, αInit=αInit,
            anticycling = anticycling, averaging = averaging, gradUp = gradUp, γ = γ, maxIter = maxIter, δ = δ)

        indices_cio, w_cio, Δt_cio, status, Gap, cutCount = oa_formulation(ℓ, Y[train], X[train, :],
                                  k, γ, indices0=result.indices, ΔT_max=TimeLimit)

        valError = error(ℓ, Y[val], X[val, :], indices_cio, w_cio)
        push!(Path, [k, indices_cio, w_cio, cutCount, γ, Δt_cio, valError])

        αInit = result.α[:]
        indInit = indices_cio[:]

        g_old = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], 0*w_cio, 2*γ))
        g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], w_cio, 2*γ))
        stop = (g_new / g_old < 1e-2)

        factor *= 2
      end
    end
  end

  return Path
end
