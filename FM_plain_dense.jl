using SparseArrays
using Statistics
using LinearAlgebra
using Distributed

struct SGDParams
    α::Float64
    num_epochs::Int64
    # regularization
    reg⁰::Float64
    regʷ::Float64
    regᵛ::Float64
end

mutable struct GaussianParams
    k₀::Bool
    k₁::Bool
    num_factors::Int64
    μ::Float64
    σ::Float64
end

mutable struct FMModel
    k₀::Bool
    k₁::Bool
    w₀::Float64
    w::Vector{Float64}
    V::Matrix{Float64}
    num_factors::Int64
end

struct RegressionTask
    target_min::Float64
    target_max::Float64
end

function predict_instance!(fm_model, idx, x)
    f_sum = zeros(fm_model.num_factors)
    sum_sqr = zeros(fm_model.num_factors)
    result = zero(Float64)
    if fm_model.k₀
        result += fm_model.w₀
    end
    if fm_model.k₁
        @inbounds for i in 1:length(idx)
            result += fm_model.w[idx[i]] * x[i]
        end
    end
    @inbounds for f in 1:fm_model.num_factors
        @inbounds for i in 1:length(idx)
            d = fm_model.V[f,idx[i]] * x[i]
            f_sum[f] += d
            sum_sqr[f] += d * d
        end
        result += 0.5 * (f_sum[f] * f_sum[f] - sum_sqr[f])
    end
    return result
end

function predict_instance!(fm_model, x)
    f_sum = zeros(fm_model.num_factors)
    sum_sqr = zeros(fm_model.num_factors)
    result = zero(Float64)
    if fm_model.k₀
        result += fm_model.w₀
    end
    if fm_model.k₁
        @inbounds for i in 1:length(x)
            result += fm_model.w[i] * x[i]
        end
    end
    @inbounds for f in 1:fm_model.num_factors
        @inbounds for i in 1:length(x)
            d = fm_model.V[f,i] * x[i]
            f_sum[f] += d
            sum_sqr[f] += d * d
        end
        result += 0.5 * (f_sum[f] * f_sum[f] - sum_sqr[f])
    end
    return result
end

X = float.(rand(Bool, 10000000, 500))#sparse(T[:,2:end]')
y = float.(rand(Bool, 10000000) .* 2 .- 1)#T[:,1]

sgd_params = SGDParams(0.01, 100, .0, .0, .0)
evaluator = x -> sqrt(mean(x))
gauss_params = GaussianParams(true, true, 10, .0, .01)

# initModel
num_samples, num_attributes = size(X)
@assert length(y) == num_samples
fm_model = FMModel(gauss_params.k₀, gauss_params.k₁, .0, zeros(num_attributes), randn(gauss_params.num_factors, num_attributes) .* gauss_params.σ .+ gauss_params.μ, gauss_params.num_factors)
fm_model_updates = FMModel(gauss_params.k₀, gauss_params.k₁, .0, zeros(num_attributes), randn(gauss_params.num_factors, num_attributes) .* gauss_params.σ .+ gauss_params.μ, gauss_params.num_factors)
# initTask
regression_task = RegressionTask(minimum(y), maximum(y))
# skipped initPredictor
XT = X'
XT_sparse = sparse(XT)
XT2 = XT .* XT

# sgd train
predictions = zeros(length(y))
diffs = zeros(length(y))
f_sum = zeros(fm_model.num_factors, length(y)) # f_sum = zeros(fm_model.num_factors, num_attributes)
@show "Learning Factorization Machines with gradient descent..."
for epoch in 1:sgd_params.num_epochs
    #@show "[SGD - Epoch $epoch] Start..."
    @time Threads.@threads for c in 1:length(y)
        XT_nzrange = nzrange(XT_sparse, c)
        idx = XT_sparse.rowval[XT_nzrange] # sub(X.rowval, X_nzrange)
        x = XT_sparse.nzval[XT_nzrange] # sub(X.nzval, X_nzrange)
        predictions[c] = max(min(predict_instance!(fm_model, idx, x), regression_task.target_max), regression_task.target_min)
        # predictions[c] = max(min(predict_instance!(fm_model, XT[:, c]), regression_task.target_max), regression_task.target_min)
        diffs[c] = predictions[c] - y[c] # loss_deriv(predictor.task, p, y[c]) = 2(p - y[c])
        f_sum[:, c] = (fm_model.V * XT[:, c]) .* (2 * diffs[c])
        # does not save time
        # f_sum .+= reshape(kron(fm_model.V * XT[:, c] .* (2 * diffs[c]), XT[:, c]), fm_model.num_factors, num_attributes)
    end
    
    #evaluation
    @time err = [diffs[i] * diffs[i] for i in 1:length(y)]
    @time evaluation = evaluator(err .* err)
    @show "[SGD - Epoch $epoch] Evaluation: $evaluation"

    # update
    if fm_model.k₀
        fm_model.w₀ -= sgd_params.α * (2 * sum(diffs) + sgd_params.reg⁰ * length(y) * fm_model.w₀)
    end
    if fm_model.k₁
        #=@time Threads.@threads for i in 1:num_attributes
            fm_model.w[i] -= sgd_params.α * (2 * dot(X[:, i], diffs) + sgd_params.regʷ * length(y) * fm_model.w[i])
        end=#
        @time fm_model.w .-= sgd_params.α .* (2 .* (XT * diffs) .+ sgd_params.regʷ * length(y) .* fm_model.w)
    end
    @time fm_model.V .-= sgd_params.α .* (f_sum * X .- fm_model.V .* (2 .* XT2 * diffs)' .+ sgd_params.regᵛ * length(y) .* fm_model.V)
    #=for c in 1:X.n # X.n = size(y)[1] = number of data points
        X_nzrange = nzrange(X, c)
        idx = X.rowval[X_nzrange] # sub(X.rowval, X_nzrange)
        x = X.nzval[X_nzrange] # sub(X.nzval, X_nzrange)
        #@show "DEBUG: processing $c"
        mult = 2 * (p - y[c]) # loss_deriv(predictor.task, p, y[c]) = 2(p - y[c])
        #@show "DEBUG: mult: $mult"
        if model.k0
            model.w0 -= alpha * (mult + sgd.reg0 * model.w0)
        end
        if model.k1
            for i in 1:length(idx)
                model.w[idx[i]] -= alpha * (mult * x[i] + sgd.regw * model.w[idx[i]])
            end
        end
        @inbounds for f in 1:fm_model.num_factors
            @inbounds for i in 1:length(idx)
                grad = f_sum[f] * x[i] - fm_model.V[f,idx[i]] * x[i] * x[i]
                fm_model.V[f,idx[i]] -= α * (mult * grad + sgd.regᵛ * fm_model.V[f,idx[i]])
            end
        end
    end=#
    #@show "[SGD - Epoch $epoch] End."
end
