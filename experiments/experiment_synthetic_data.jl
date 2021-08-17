using DataFrames, CSV

using GLMNet, Distributions, Random
using SubsetSelection
using SubsetSelectionCIO
include("data_construction.jl") #Functions to generate data
include("cv.jl")    #Utilities to conduct cross-validation
include("oa/oa_cv.jl")


EXPERIMENTS = DataFrame(prefix=["ML"], snr = [1e8], ρ = [.3])



## Choose loss function for sparse classication -- See SubsetSelection.jl documentation
ℓ = SubsetSelection.L1SVM()
#ℓ = SubsetSelection.LogReg()

do_lasso = false #Indicates whether to compute the Lasso solution
do_enet = false #Indicates whether to compute the Enet solution
do_saddle = true #Indicates whether to use the Boolean relaxation as warmstart to CIOF
do_cio = true #Indicates whether to compute the CIO solution

savedir = "results/"
mkpath(savedir)

iter_run = 1 #Number of the iteration (for averaging)
n = 500 #Number of observations
p = 1000 #Number of features
k = 30 #Size of the support


prefix= "ML" #prefix for result files
snr = 1e8 #signal-to-noise ratio
ρ = 0.3 #correlation in the Toeplitz correlation matrix


#Dataframes to save results
LASSO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                time=Real[], A=Real[], AUC=Real[], MR=Real[])
ENET = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                alpha=Real[], time=Real[], A=Real[], AUC=Real[], MR=Real[])
SADDLE = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                gamma=Real[], factor=[], time=Real[], cuts=Real[], A=Real[], AUC=Real[], MR=Real[])
CIO = DataFrame(run = Int[], n=Int[], r=Int[], SNR=Real[], rho= Real[], k=Int[],
                gamma=Real[], factor=[], time=Real[], Gap=Real[], cuts=Real[], A=Real[], AUC=Real[], MR=Real[])


Random.seed!(n+536*iter_run)


println("***********************")
println("BUILDING DATA")
println("***********************")
X, Y, indices_true, w_true = data_construction(SubsetSelection.L1SVM(), n+200, k, p, snr, ρ)

normalization = sqrt.(sum(X.^2,dims=1))/sqrt.(size(X,1))
X = X./normalization

train = 1:n
val = (n+1):(n+200)

println("***********************")
println("Samples = ", n)
println("Features = ", p)
println("Sparsity = ", k)
println("SNR = ", snr)
println("ρ = ", ρ)
println("***********************")


println("***********************")
println("Julia-LASSO")
println("***********************")
Y_transf = [(Y[train].<0) (Y[train].>0)]

t0 = time_ns()
l1 = glmnet(X[train,:], convert(Matrix{Float64}, Y_transf), GLMNet.Binomial(), intercept=false)
Δt_lasso = (time_ns() - t0)*(1e-9)

if do_lasso
    col = size(l1.betas, 2)
    while length(findall(s->abs(s)>1e-8, l1.betas[:, col]))>k && col>=1
        col = col - 1
    end

    indices_lasso = findall(s->abs(s)>1e-8, l1.betas[:, col])
    w_lasso = l1.betas[:, col][indices_lasso]

    accuracy_lasso = length(intersect(indices_true, indices_lasso))/k*100
    auc_lasso = 1-error(ℓ, Y[val], X[val,:], indices_lasso, w_lasso)
    mr_lasso = error2(ℓ, Y[val], X[val,:], indices_lasso, w_lasso)
    push!(LASSO, [iter_run, n, p, snr, ρ, k, Δt_lasso, accuracy_lasso, auc_lasso, mr_lasso])
    filename = string(prefix, "LassoFix", array_num, ".csv")
    CSV.write(savedir*filename, LASSO)
end

println("***********************")
println("Julia-ENET")
println("***********************")
if do_enet
    for α in 0.1:.1:1
        t0 =time_ns()
        enet = glmnet(X[train,:], convert(Matrix{Float64}, Y_transf), GLMNet.Binomial(), intercept=false, alpha=α)
        Δt_enet = (time_ns() - t0)*(1e-9)

        col = size(enet.betas, 2)
        while length(findall(s->abs(s)>1e-8, enet.betas[:, col]))>k && col>=1
            col = col - 1
        end

        indices_enet = findall(s->abs(s)>1e-8, enet.betas[:, col])
        w_enet = enet.betas[:, col][indices_enet]

        accuracy_enet = length(intersect(indices_true, indices_enet))/k*100
        auc_enet = 1-error(ℓ, Y[val], X[val,:], indices_enet, w_enet)
        mr_enet = error2(ℓ, Y[val], X[val,:], indices_enet, w_enet)
        push!(ENET, [iter_run, n, p, snr, ρ, k, α, Δt_enet/Δt_lasso, accuracy_enet, auc_enet, mr_enet])
        filename = string(prefix, "EnetFix", array_num, ".csv")
        CSV.write(savedir*filename, ENET)
    end
end


println("***********************")
println("Julia - SADDLE POINT RELAXATION and CUTTING PLANES")
println("***********************")
if do_saddle||do_cio
  # Regularization
  subsetSelection(ℓ, Constraint(10), Y[1:10], X[1:10,:], maxIter=10)

  γ0 = 1. *p / k / (maximum(sum(X[train,:].^2,dims=2))*n)
  factor = 1000.

  indicesRelax=findall(x-> x< k/size(X,2), rand(size(X,2)))
  stop = false;

  for inner_epoch in 1:10

      if !stop
        γ = factor*γ0
        println("***********************")
        println("Sparsity k = ", k)
        println("Regularization γ = ", γ)
        println("***********************")

        #Initialization for CIO: Either Boolean relaxation or GLMNet
        if do_saddle
            maxCut = 100
            t0=time_ns()
            result_saddle = subsetSelection(SubsetSelection.L1SVM(), Constraint(k), Y[train], X[train,:],
                                            intercept=false, numberRestarts=1, maxIter=maxCut)
            Δt_saddle = (time_ns() - t0)*(1e-9)
            indices_saddle = result_saddle.indices[:]
            w_saddle = result_saddle.w

            accuracy_saddle = length(intersect(indices_true, indices_saddle))/k*100
            auc_saddle = 1-error(SubsetSelection.L1SVM(), Y[val], X[val,:], indices_saddle, w_saddle)
            mr_saddle = error2(SubsetSelection.L1SVM(), Y[val], X[val,:], indices_saddle, w_saddle)

            push!(SADDLE, [iter_run, n, p, snr, ρ, k,
                γ, factor, Δt_saddle/Δt_lasso, maxCut, accuracy_saddle, auc_saddle, mr_saddle])
            filename = string(prefix, "SaddleFix", array_num, ".csv")
            CSV.write(savedir*filename, SADDLE)

            indicesRelax = result_saddle.indices[:]
        else
            l1 = glmnet(X[train,:], convert(Matrix{Float64}, Y_transf), GLMNet.Binomial(), dfmax=k, intercept=false)
            col = size(l1.betas, 2)
            while length(findall(s->abs(s)>1e-8, l1.betas[:, col]))>k && col>=1
                col = col - 1
            end
            indicesRelax = findall(s->abs(s)>1e-8, l1.betas[:, col])
        end

        indices_cio, w_cio, Δt_cio, status, Gap, cutCount = oa_formulation(ℓ, Y[train], X[train,:], k, γ,
              indices0=indicesRelax,
              solver=:Gurobi, ΔT_max=180., verbose=false, rootnode=true, rootCuts=100, stochastic=true)

        accuracy_cio = length(intersect(indices_true, indices_cio))/k*100
        auc_cio = 1-error(ℓ, Y[val], X[val,:], indices_cio, w_cio)
        mr_cio = error2(ℓ, Y[val], X[val,:], indices_cio, w_cio)

        push!(CIO, [iter_run, n, p, snr, ρ, k,
            γ, factor, Δt_cio, Gap, cutCount, accuracy_cio, auc_cio, mr_cio])
        filename = string(prefix, "CIOFix", array_num, "_StochasticWS.csv")
        CSV.write(savedir*filename, CIO)

        g_old = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], zeros(length(indices_cio)), 2*γ))
        g_new = norm(grad_primal(ℓ, Y[train], X[train, indices_cio], w_cio, 2*γ))
        stop = (g_new / g_old < 1e-2)

        factor *=2
      end
  end
end
