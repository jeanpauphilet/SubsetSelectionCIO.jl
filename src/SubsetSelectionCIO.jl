module SubsetSelectionCIO

using SubsetSelection, JuMP, Gurobi, CPLEX, MathOptInterface, LinearAlgebra, GLMNet

import Compat.String

include("inner_op.jl")

export oa_formulation

getthreads() = haskey(ENV, "SLURM_JOB_CPUS_PER_NODE") ? parse(Int, ENV["SLURM_JOB_CPUS_PER_NODE"]) : 0

###########################
# FUNCTION oa_formulation
###########################
"""Computes the minimum regression error with Ridge regularization subject an explicit
cardinality constraint using cutting-planes.

w^* := arg min  ∑_i ℓ(y_i, x_i^T w) +1/(2γ) ||w||^2
           st.  ||w||_0 = k

INPUTS
  ℓ           - LossFunction to use
  Y           - Vector of outputs. For classification, use ±1 labels
  X           - Array of inputs
  k           - Sparsity parameter
  γ           - ℓ2-regularization parameter
  indices0    - (optional) Initial solution
  ΔT_max      - (optional) Maximum running time in seconds for the MIP solver. Default is 60s
  gap         - (optional) MIP solver accuracy

OUTPUT
  indices     - Indicates which features are used as regressors
  w           - Regression coefficients
  Δt          - Computational time (in seconds)
  status      - Solver status at termination
  Gap         - Optimality gap at termination
  cutCount    - Number of cuts needed in the cutting-plane algorithm
  """
function oa_formulation(ℓ::LossFunction, Y, X, k::Int, γ;
          indices0=findall(rand(size(X,2)) .< k/size(X,2)),
          ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi,
          rootnode::Bool=true, stochastic::Bool=false)

  n,p = size(X)

  miop = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(CPLEX.Optimizer)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", ΔT_max)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", Gap)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "Threads" : "CPXPARAM_Threads", getthreads())

  s0 = zeros(p); s0[indices0] .= 1.
  c0, ∇c0 = inner_op(ℓ, Y, X, s0, γ, stochastic=stochastic)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=1.005*c0)

  for j in 1:p
    JuMP.set_start_value(s[j], s0[j])
  end
  JuMP.set_start_value(t, 1.005*c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s) <= k)

  #Root node analysis
  cutCount=1; bestObj=sum(s0)<= k ? c0 : +Inf; bestSolution=sum(s0)<= k ? s0[:] : [] ;
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  if rootnode
    s1 = zeros(p)
    l1 = isa(ℓ, SubsetSelection.Classification) ? glmnet(X, convert(Matrix{Float64}, [(Y.<0) (Y.>0)]), GLMNet.Binomial(), dfmax=k, intercept=false) : glmnet(X, Y, dfmax=k, intercept=false)
    for  i in size(l1.betas, 2):-1:max(size(l1.betas, 2)-20,1) #Add first 20 cuts from Lasso path
      ind = findall(abs.(l1.betas[:, i]) .> 1e-8); s1[ind] .= 1.
      c1, ∇c1 = inner_op(ℓ, Y, X, s1, γ, stochastic=stochastic)
      @constraint(miop, t>= c1 + dot(∇c1, s-s1))
      cutCount += 1; s1 .= 0.
    end
  end

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb_data)
    s_val = [callback_value(cb_data, s[j]) for j in 1:p] #vectorized version of callback_value is not currently offered in JuMP
    # if maximum(s_val.*(1 .- s_val)) < 10*eps()
      s_val = 1.0 .* (rand(p) .< s_val) #JuMP updates calls Lazy Callbacks at fractional solutions as well

      c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=stochastic)
      if stochastic && callback_value(cb_data, t) > c #If stochastic version and did not cut the solution
          c, ∇c = inner_op(ℓ, Y, X, s_val, γ, stochastic=false)
      end
      if sum(s_val)<=k && c<bestObj #if feasible and best value
        bestObj = c; bestSolution=s_val[:]
      end

      con = @build_constraint(t >= c + dot(∇c, s-s_val))
      MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
      cutCount += 1
    # end
  end
  MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

  # # Feed warmstart
  # if (solver == :Gurobi)
  #   wsCount = 0
  #   function warm_start(cb_data)
  #     if wsCount == 0
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [s[j] for j in 1:p], floor.(Int, s0))
  #       MOI.submit(miop, MOI.HeuristicSolution(cb_data), [t], [c0])
  #       wsCount += 1
  #     end
  #   end
  #   MOI.set(miop, MOI.HeuristicCallback(), warm_start)
  # end

  @show bestObj, sum(bestSolution)

  t0 = time()
  optimize!(miop)
  status = termination_status(miop)
  Δt = JuMP.solve_time(miop)

  Gap = 1 - JuMP.objective_bound(miop) /  abs(JuMP.objective_value(miop))
  if JuMP.objective_bound(miop) < bestObj
    bestSolution = value.(s)[:]
  end

  # Find selected regressors and run a standard linear regression with Tikhonov regularization
  indices = findall(bestSolution .> .5)
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  return indices, w, Δt, status, Gap, cutCount
end

end # module
