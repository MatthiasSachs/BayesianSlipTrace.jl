
using BayesianSlipTrace
using Distributions
using StatsPlots
using Random

# Set a random seed.
Random.seed!(3)

# Define Gaussian mixture model.
σ = 0.2
w = [0.5, 0.5]
μ = [.5, π-.1]
K = 2
mixturemodel = MixtureModel([PeriodicNormal(μₖ,σ, Float64(π)) for μₖ in μ], w)
# We draw the data points.
N = 5000
x = rand(mixturemodel, N);

#%%
using Turing

@model function periodic_gaussian_mixture_model(x,K,μ)
    w ~ Dirichlet(K, 1.0)
    mixturemodel = MixtureModel([PeriodicNormal(μₖ,σ,Float64(π)) for μₖ in μ], w)
    for i in 1:N
        x[i] ~ mixturemodel
    end

    return x
end

model = periodic_gaussian_mixture_model(x,K,μ);

#%%
sampler = HMC(0.005, 10)
nsamples = 500
nchains = 10
burn = 10
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains, discard_initial = burn, progress = true)

#%%
chains
plot(chains[["w[1]", "w[2]"]]; legend=true)


