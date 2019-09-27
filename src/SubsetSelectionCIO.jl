module SubsetSelectionCIO

using MathProgBase, JuMP, Gurobi, CPLEX, Compat, LinearAlgebra
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
          indices0=findall(x-> x<k/size(X,2), rand(size(X,2))), ΔT_max=60, verbose=false, Gap=0e-3, solver::Symbol=:Gurobi)

  n,p = size(X)

  miop = (solver == :Gurobi) ?    Model(solver=GurobiSolver(MIPGap=Gap, TimeLimit=ΔT_max,
                                    OutputFlag=1*verbose, LazyConstraints=1, Threads=getthreads())) :
                                  Model(solver=CplexSolver(CPX_PARAM_EPGAP=Gap, CPX_PARAM_TILIM=ΔT_max,
                                    CPX_PARAM_SCRIND=1*verbose))
  s0 = zeros(p); s0[indices0].=1

  c0, ∇c0 = inner_op(ℓ, Y, X, s0, γ)

  # Optimization variables
  @variable(miop, s[j=1:p], Bin, start=s0[j])
  @variable(miop, t>=0, start=c0)

  # Objective
  @objective(miop, Min, t)

  # Constraints
  @constraint(miop, sum(s)<=k)

  cutCount=1; bestObj=c0; bestSolution=s0[:];
  @constraint(miop, t>= c0 + dot(∇c0, s-s0))

  #Info array
  bbdata = DataFrame(time=Float64[], node=Int[], obj=Float64[], bestbound=Float64[])

  # Outer approximation method for Convex Integer Optimization (CIO)
  function outer_approximation(cb)
    cutCount += 1
    c, ∇c = inner_op(ℓ, Y, X, getvalue(s), γ)
    if c<bestObj
      bestObj = c; bestSolution=getvalue(s)[:]
    end

    node      = MathProgBase.cbgetexplorednodes(cb)
    obj       = bestObj
    bestbound = MathProgBase.cbgetbestbound(cb)
    push!(bbdata, [time()-t0,node,obj,bestbound])

    @lazyconstraint(cb, t>=c + dot(∇c, s-getvalue(s)))
  end
  addlazycallback(miop, outer_approximation)

  t0=time()
  status = solve(miop)
  Δt = getsolvetime(miop)

  if status != :Optimal
    Gap = 1 - JuMP.getobjbound(miop) /  getobjectivevalue(miop)
  end
  if status == :Optimal
    bestSolution = getvalue(s)[:]
  end
  # Find selected regressors and run a standard linear regression with Tikhonov regularization
  indices = findall(s->s>0.5, bestSolution)
  w = SubsetSelection.recover_primal(ℓ, Y, X[:, indices], γ)

  return indices, w, Δt, status, Gap, cutCount
end

end # module
