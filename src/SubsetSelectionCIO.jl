module SubsetSelectionCIO

using SubsetSelection, JuMP, Gurobi, MathOptInterface, LinearAlgebra
using DataFrames, CSV

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
          indices0=findall(rand(size(X,2)) .< k/size(X,2)), ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi)

  n,p = size(X)

  miop = (solver == :Gurobi) ? Model(Gurobi.Optimizer) : Model(Cplex.Optimizer)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "TimeLimit" : "CPX_PARAM_TILIM", ΔT_max)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "OutputFlag" : "CPX_PARAM_SCRIND", 1*verbose)
  set_optimizer_attribute(miop, (solver == :Gurobi) ? "MIPGap" : "CPX_PARAM_EPGAP", Gap)

  s0 = zeros(p); s0[indices0] .= 1.
  c0, ∇c0 = inner_op(ℓ, Y, X, s0, γ)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s) <= k)

  cutCount=1; bestObj=c0; bestSolution=s0[:];
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb_data)
    s_val = [callback_value(cb_data, s[j]) for j in 1:p] #vectorized version of callback_value is not currently offered in JuMP
    c, ∇c = inner_op(ℓ, Y, X, s_val, γ)
    if c<bestObj
      bestObj = c; bestSolution=s_val[:]
    end

    con = @build_constraint(t >= c + dot(∇c, s-s_val))
    MOI.submit(miop, MOI.LazyConstraint(cb_data), con)
    cutCount += 1
  end
  MOI.set(miop, MOI.LazyConstraintCallback(), outer_approximation)

  t0 = time()
  status = optimize!(miop)
  Δt = JuMP.solve_time(miop)

  if status != :Optimal
    Gap = 1 - JuMP.objective_bound(miop) /  JuMP.objective_value(miop)
  end
  if status == :Optimal
  b estSolution = value(s)[:]
  end
  # Find selected regressors and run a standard linear regression with Tikhonov regularization
  indices = findall(bestSolution .> .5)
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  return indices, w, Δt, status, Gap, cutCount
end

end # module
