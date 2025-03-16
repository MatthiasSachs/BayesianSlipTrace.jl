module BayesianSlipTrace


using Random
using Distributions
using Turing

export PeriodicNormal
export sp_uinfl_mix
export sp_mix

struct PeriodicNormal{T<:Real} <: ContinuousUnivariateDistribution 
    mm::Normal{T}
    μ::T
    b::T
end

PeriodicNormal(μ::T, σ::T, b::T) where {T<:Real} = PeriodicNormal(Normal(0.0, σ), μ, b)

function Distributions.rand(rng::AbstractRNG, d::PeriodicNormal)
    return mod(Distributions.rand(rng, d.mm) + d.μ, d.b)
end

function Distributions.logpdf(d::PeriodicNormal, x::Real)
    dx = x - d.μ
    if dx > d.b * 0.5 
        return Distributions.logpdf(d.mm,dx - d.b)
    elseif dx <= -d.b * 0.5
        return Distributions.logpdf(d.mm,dx + d.b)
    else
        return Distributions.logpdf(d.mm,dx)
    end

end


@model function sp_mix(X, Nu, N_img, K, σ)
    w ~ Dirichlet(K, 1.0)
    for i in 1:N_img
        for j in 1:length(X[i])
            X[i][j] ~ MixtureModel([MixtureModel([PeriodicNormal(ν, σ,Float64(π)) for ν in Nu[i][k]], ones(length(Nu[i][k]))/length(Nu[i][k])) for k = 1:K],w)
        end
    end
    return X
end


@model function sp_uinfl_mix(X, Nu, N_img, K, σ)
    w ~ Dirichlet(K+1, 1.0)
    for i in 1:N_img
        for j in 1:length(X[i])
            X[i][j] ~ MixtureModel(vcat([MixtureModel([PeriodicNormal(ν, σ,Float64(π)) for ν in Nu[i][k]], ones(length(Nu[i][k]))/length(Nu[i][k])) for k = 1:K],[Uniform(0.0,Float64(π))]),w)
        end
    end
    return X
end


end