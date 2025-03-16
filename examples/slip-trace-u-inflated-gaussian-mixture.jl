using BayesianSlipTrace
using Distributions
using FillArrays
using StatsPlots
using LinearAlgebra
using Random
using Turing
# Set a random seed.
Random.seed!(3)

# Generate synthetic data
σ = 0.01
K = 4 
w_true = rand(Dirichlet(ones(K+1))) # true mixture weights
number_of_angles = rand(3:5,K) # number of angles in each component

X = Vector{Vector{Float64}}()
Nu = Vector{Vector{Vector{Float64}}}()
N_img = 20 # number of images
for i in 1:N_img
    ν_vecs = [rand(Uniform(0,π), k) for k in number_of_angles] # true means; these would need to be replaced by the range of possible angles 
    number_of_lines = rand(10:30) # number of lines in each image
    x = rand(MixtureModel( vcat([MixtureModel([PeriodicNormal(ν, σ,Float64(π)) for ν in ν_vecs[k]], ones(number_of_angles[k])/number_of_angles[k]) for k = 1:K],[Uniform(0.0,Float64(π))]) ,w_true), number_of_lines)
    push!(X, x)
    push!(Nu, ν_vecs)
end


#%%
using BayesianSlipTrace: sp_uinfl_mix
model = sp_uinfl_mix(X, Nu, N_img, K,σ)

#%%
# Sample from the posterior distribution
sampler = HMC(0.05, 10) # Hamiltonian Monte Carlo sampling with a stepsize of 0.05 and 10 leapfrog steps, make sure that the acceptance rate is somewhere between 0.3 and 0.9.
nsamples = 5000
nchains = 10
burn = 10
chains = sample(model, sampler, MCMCThreads(), nsamples, nchains, discard_initial = burn, progress = true);

chains # Check that the effective sample size (ESS) is sufficiently large. Adapt number of samples and stepsize if neccesary 
#%%
chains
plot(chains[["w[1]", "w[2]", "w[3]", "w[4]","w[5]"]]; legend=true)


ss = summarystats(chains)


#%%
# Plot marginal posterior probability distributions of the weight parameters
pl = ridgelineplot(chains,Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"]);hpd_val = [0.05, 0.2],
q = [0.1, 0.9],
spacer = 15.0,
_riser = 0.1,
show_mean = true,
show_median = true,
show_qi = false,
show_hpdi = true,
fill_q = true,
fill_hpd = false,
ordered =false,
xlims = (0.0, 1.0),
size=(500,500)
)
ylabel!(pl,"Weights")
display(pl)
#StatsPlots.savefig(pl,joinpath("", "./plots/demo/ridge_plot_.pdf"))
#%%

forestplot(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"]), hpd_val = [0.05, 0.15, 0.25], size=(500,500))


# Other useful visualiztations of the Markov chain samples:
display(meanplot(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"])))
display(traceplot(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"])))
display(autocorplot(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"])))

display(histogram(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"]),append_chains=true ))
display(density(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"]),append_chains=true ))

# the corner plot does not seem to work particularly well. Use the corrplot below instead. (Or export the data to e.g. and generate this kind of plot in R or python (using e.g. Seaborn))
display(corner(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"]),
xlims = (0.0, 1.0),
ylims = (0.0, 1.0), 
append_chains=true
))


display(density(chains, Symbol.(["w[1]", "w[2]", "w[3]", "w[4]"])))

#%%

#%%
# Custom plot of the marginal posterior mean, median and 95% credibility 
# intervals for the weight parameters compared to the true weight values. Requires PyPlot to run!
using PyPlot
mz = 6
lw = 1.5
alpha = 0.05
mhpd = hpd(chains;alpha=alpha)
ss = summarystats(chains)
qs = quantile(chains, q = [0.025, 0.25, 0.5, 0.75, 0.975])
figfactor = 2
fig = figure()
lo,up = mhpd[:,:upper],mhpd[:,:lower]
for i = 1:(K+1)
    PyPlot.plot([i,i], [lo[i],up[i]], "b-", lw=lw)
end
PyPlot.plot(1:(K+1), ss[:,:mean],"ro", markersize=mz, label="posterior mean" )
PyPlot.plot(1:(K+1), qs[:,Symbol("50.0%")],"bx", markersize=mz, label="posterior median" )
PyPlot.plot(collect(1:K+1), w_true, "gx", markersize=mz, label="true parameter")
# PyPlot.grid(true)
xlabel("Component index")
ylabel("Value")
PyPlot.ylim(0,1)
PyPlot.xticks(collect(1:K))
title("$(100*(1-alpha))% Credibility intervals for regressors")
legend()
display(gcf())

#%%
# Export the samples to a DataFrame (useful for further processing or exporting.)
using DataFrames
mdf = DataFrame(chains)

# Plot correlation plots of the 2d marginals of the posterior distribution of the weight parameters
@df mdf[:,["w[1]","w[2]","w[3]", "w[4]"]] corrplot(cols(1:4), grid = true, size=(800,800))


#%%
# Export the samples to a matrix (useful for further processing or exporting.) 
A = Array(chains, append_chains=true)

marginalkde(A[:,1], A[:,2])
cornerplot(A, compact=true,xlims = (0.0, 1.0), ylims = (0.0, 1.0))

