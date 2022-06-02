using SparseArrays
using Statistics
using Random
using LinearAlgebra
using MLDataUtils

abstract type MethodParams
end

abstract type Evaluator
end

abstract type ModelParams
end

abstract type TaskParams
end

struct SquaredErrorEvaluator <: Evaluator
    stat::Function
    SquaredErrorEvaluator(; stat::Function = x -> sqrt(mean(x))) = new(stat)
end
const rmse = SquaredErrorEvaluator

"""Heaviside step function"""
heaviside(p::Number, y::Number) = p == y ? zero(p) : one(p)

"""Sigmoid"""
sigmoid(x::Number) = one(x) / (one(x) + exp(-x))

"""Squared error"""
sqerr(p::Number, y::Number)       = (p - y)^2
sqerr_deriv(p::Number, y::Number) = 2(p - y)

"""Negative logistic sigmoid"""
nlogsig(p::Number, y::Number)       = -log(sigmoid(p * y))
nlogsig_deriv(p::Number, y::Number) = y * sigmoid(p * y) - y

"""Binomial cross entropy"""
bce(sp::Number, y::Number) = -y * log(sp) - (1 - y) * (sp > (1 - 10e-9) ? 10e-9 : log(1 - sp))
bce_deriv(sp::Number, y::Number) = sp - (1 - y)

"""Represents a classification task"""
struct ClassificationTaskParams <: TaskParams
end
const classification = ClassificationTaskParams

"""Represents a regression task"""
struct RegressionTaskParams <: TaskParams
end
const regression = RegressionTaskParams

abstract type PredictorTask
end

"""Classification parameters derived from data"""
struct ClassificationTask <: PredictorTask
end

loss(::ClassificationTask, p::Number, y::Number)       = bce(p, y)
loss_deriv(::ClassificationTask, p::Number, y::Number) = bce_deriv(p, y)

"""Regression parameters derived from data"""
struct RegressionTask <: PredictorTask
    target_min::Float64
    target_max::Float64

    RegressionTask(; target_min::Float64 = typemin(Float64), target_max::Float64 = typemax(Float64)) =
        new(target_min, target_max)
end

loss(::RegressionTask, p::Number, y::Number)       = sqerr(p, y)
loss_deriv(::RegressionTask, p::Number, y::Number) = sqerr_deriv(p, y)

mutable struct GaussianModelParams <: ModelParams
    k₀::Bool
    k₁::Bool
    num_factors::Int64
    μ::Float64
    σ::Float64
    GaussianModelParams(; k₀ = true, k₁ = true, num_factors = 8, μ = .0, σ = .01) =
        new(k₀, k₁, num_factors, μ, σ)
end
const gauss = GaussianModelParams

mutable struct FMModel
    k₀::Bool
    k₁::Bool
    b::Float64
    u::Vector{Float64}
    V::Matrix{Float64}
    num_factors::Int64
end

struct FMPredictor{T<:PredictorTask}
    task::T
    model::FMModel
    model_Δ::FMModel
end

struct SGDMethod <: MethodParams
    α::Float64 # learning rate
    γ::Float64 # momentum
    num_epochs::Int64
    # regularization
    λ₀::Float64
    λᵤ::Float64
    λᵥ::Float64
    SGDMethod(; α::Float64 = 0.01, γ::Float64 = 0.9, num_epochs::Int64 = 100, λ₀::Float64 = .0, λᵤ::Float64 = .0, λᵥ::Float64 = .0) =
        new(α, γ, num_epochs, λ₀, λᵤ, λᵥ)
end
const sgd = SGDMethod

function read_libsvm(fname::String, dimension = :col)
    label = Float64[]
    mI = Int64[]
    mJ = Int64[]
    mV = Float64[]
    fi = open(fname, "r")
    cnt = 1
    for line in eachline(fi)
        line = split(strip(line), " ")
        push!(label, parse(Float64, line[1]))
        line = line[2:end]
        for itm in line
            itm = split(itm, ":")
            push!(mI, parse(Int, itm[1]) + 1)
            push!(mJ, cnt)
            push!(mV, parse(Float64, itm[2]))
        end
        cnt += 1
    end
    close(fi)

    if dimension == :col 
        (sparse(mI,mJ,mV), label)
    else
        (sparse(mJ,mI,mV), label)
    end
end

function roc_auc(y, y′, intervals = 100)
    @assert length(y) == length(y′)

    auc = 0.0
    TPR, FPR = zeros(Float64, intervals), zeros(Float64, intervals)
    for i in 1:intervals
        TP, FN, FP, TN = 0, 0, 0, 0
        for j in 1:length(y)
            if y[j] > 0 # must be either 0 or 1
                if y′[j] >= i / intervals # sigmoid within 0 and 1
                    TP += 1
                else
                    FN += 1
                end
            else
                if y′[j] >= i / intervals
                    FP += 1
                else
                    TN += 1
                end
            end
        end
        TPR[i] = TP / (TP + FN)
        FPR[i] = FP / (TN + FP)
        if i > 1
            auc += (FPR[i - 1] - FPR[i]) * (TPR[i] + TPR[i - 1]) / 2
        end
    end
    auc, TPR, FPR
end

function initModel(params::GaussianModelParams, X::SparseMatrixCSC, y::Vector{Float64})
    # initialization
    num_samples, num_attributes = size(X)
    # sanity check
    @assert length(y) == num_samples

    # create initial model
    Random.seed!(1234)
    b = .0
    u = zeros(num_attributes)
    V = randn(num_attributes, params.num_factors) .* params.σ .+ params.μ
    b = 0.2098
    u = [0.3174; 0.3704; -0.2549]
    V = [0.0461 0.4024; -1.0115 0.2167; -0.6123  0.5036]

    # new model
    # return (FMModel(params.k₀, params.k₁, b, u, V, params.num_factors), FMModel(params.k₀, params.k₁, b, u, zeros(num_attributes, params.num_factors), params.num_factors))
    return (FMModel(params.k₀, params.k₁, b, u, V, params.num_factors), FMModel(params.k₀, params.k₁, 0, zeros(num_attributes), zeros(num_attributes, params.num_factors), params.num_factors))
end

"""
Given data `X` and `y`, initializes a `ClassificationTask`
"""
function initTask(::ClassificationTaskParams, X::SparseMatrixCSC, y::Vector{Float64})
    ClassificationTask()
end

"""
Given data `X` and `y`, initializes a `RegressionTask`
"""
function initTask(::RegressionTaskParams, X::SparseMatrixCSC, y::Vector{Float64})
    RegressionTask(target_min = minimum(y), target_max = maximum(y))
end

function predict_instance!(model::FMModel,
    idx::StridedVector{Int64}, x::StridedVector{Float64},
    f_sum::Vector{Float64}, sum_sqr::Vector{Float64})

    fill!(f_sum, .0)
    fill!(sum_sqr, .0)
    result = zero(Float64)
    if model.k₀
        result += model.b
    end
    if model.k₁
        for i in 1:length(idx)
            result += model.u[idx[i]] * x[i]
        end
    end
    @inbounds for f in 1:model.num_factors
        @inbounds for i in 1:length(idx)
            d = model.V[f,idx[i]] * x[i]
            f_sum[f] += d
            sum_sqr[f] += d * d
        end
        result += 0.5 * (f_sum[f] * f_sum[f] - sum_sqr[f])
    end
    result
end

"""Instance prediction specialized for classification or regression"""
function predict_instance!(predictor::FMPredictor,
                           idx::StridedVector{Int64}, x::StridedVector{Float64},
                           f_sum::Vector{Float64}, sum_sqr::Vector{Float64})

    if typeof(predictor.task) == ClassificationTask
        p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
        sigmoid(-p)
    else
        p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
        max(min(p, predictor.task.target_max), predictor.task.target_min)
    end
end

function sgd_update!(
    sgd::SGDMethod, model::FMModel, model_Δ::FMModel,
    X::SparseMatrixCSC,
    total_losses::Array{Float64}, cross_terms::Matrix{Float64})

    if model.k₀
        curr = model.b
        model.b -= sgd.α * (-sum(total_losses) / X.m + sgd.γ * model_Δ.b + sgd.λ₀ * model.b)
        model_Δ.b = curr - model.b
    end

    if model.k₁
        curru = copy(model.u)
        model.u .-= sgd.α .* (-X' * total_losses ./ X.m .+ sgd.γ * model_Δ.u .+ sgd.λᵤ .* model.u)
        model_Δ.u = curru .- model.u
        #=for i in 1:length(model.u)
            curr = model.u[i]
            model.u[i] -= sgd.α .* (-X[:, i]' * total_losses ./ X.m .+ sgd.γ * model_Δ.u[i] .+ sgd.λᵤ .* model.u[i])
            model_Δ[i] = curr - model.u[i]
        end=#
    end

    x_loss_terms = X .* total_losses ./ X.m
    currV = copy(model.V)
    @inbounds for f in 1:model.num_factors
        Δ = zeros(X.n)
        @inbounds for i in 1:X.n # cross_terms = X * model.V
            Δ[i] = dot(cross_terms[:, f] .- X[:, i] .* model.V[i, f], -x_loss_terms[:, i])
        end
        model.V[:, f] .-= sgd.α .* (Δ .+ sgd.γ * model_Δ.V[:, f] .+ sgd.λᵥ .* model.V[:, f])
    end
    model_Δ.V = currV .- model.V
    #=@inbounds for f in 1:model.num_factors
        currV = model.V[:, f]
        Δ = zeros(X.n)
        @inbounds for i in 1:X.n # cross_terms = X * model.V
            Δ[i] = dot(cross_terms[:, f] .- X[:, i] .* model.V[i, f], -x_loss_terms[:, i])
        end
        model.V[:, f] .-= sgd.α .* (Δ .+ sgd.γ * model_Δ.V[:, f] .+ sgd.λᵥ .* model.V[:, f])
        model_Δ.V[:, f] = currV .- model.V[:, f]
    end=#
end

function sgd_epoch!(
    sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor,
    X::SparseMatrixCSC, y::StridedVector{Float64}, epoch::Int64)

    #=total_losses = zeros(Float64, X.n)
    for c in 1:X.n # X.n = size(y)[1] = number of data points
        X_nzrange = nzrange(X, c)
        x = X.nzval[X_nzrange]
        #@show "DEBUG: processing $c"
        predictions[c] = sigmoid(-predictor.model.b - dot(predictor.model.u, x) - sum((predictor.model.V * x) .^ 2 - model.V.^2 * x.^2) / 2)
        #@show "DEBUG: prediction: $predictions[c]"
        total_losses[c] = loss_deriv(predictor.task, predictions[c], y[c])
        #@show "DEBUG: total loss: $total_losses[c]"
    end=#
    cross_terms = X * predictor.model.V
    predictions = sigmoid.(-predictor.model.b .- X * predictor.model.u .- sum(cross_terms .^ 2 .- X.^2 * predictor.model.V.^2, dims = 2) ./ 2)
    total_losses = loss_deriv.(fill(predictor.task, X.m), predictions, y)
    # batch update
    sgd_update!(sgd, predictor.model, predictor.model_Δ, X, total_losses, cross_terms)
    #evaluation
    # @time evaluation = evaluate!(evaluator, predictor, X, y, predictions)
    # err = [sqerr(predictions[i], y[i]) for i in 1:length(y)]
    err = bce.(predictions, y)
    evaluation = evaluator.stat(err .* err)
    @show "[SGD - Epoch $epoch] Evaluation: $evaluation"
end

function sgd_train!(
    sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor,
    X::SparseMatrixCSC, y::StridedVector{Float64})

    @show "Learning Factorization Machines with gradient descent..."
    for epoch in 1:sgd.num_epochs
        #@show "[SGD - Epoch $epoch] Start..."
        @time sgd_epoch!(sgd, evaluator, predictor,
                         X, y, epoch)
        #@show "[SGD - Epoch $epoch] End."
    end
end

function train(X::SparseMatrixCSC, y::Vector{Float64};
    method::SGDMethod         = sgd(α = 1.0, γ = 1.0, num_epochs = 3, λ₀ = .0, λᵤ = .0, λᵥ = .0),
    evaluator::Evaluator      = rmse(),
    task_params::TaskParams   = classification(),
    model_params::ModelParams = gauss(k₀ = true, k₁ = true, num_factors = 2, μ = .0, σ = 1.0))

    (model, model_Δ) = @time initModel(model_params, X, y)    
    task = @time initTask(task_params, X, y)
    predictor = @time FMPredictor(task, model, model_Δ)

    # Train the predictor using SGD
    sgd_train!(method, evaluator, predictor, X, y)

    predictor
end

function train_fold(X::SparseMatrixCSC, y::Vector{Float64};
    method::SGDMethod         = sgd(α = 1.0, γ = 1.0, num_epochs = 3, λ₀ = .0, λᵤ = .0, λᵥ = .0),
    evaluator::Evaluator      = rmse(),
    task_params::TaskParams   = classification(),
    model_params::ModelParams = gauss(k₀ = true, k₁ = true, num_factors = 2, μ = .0, σ = 1.0),
    kₙ::Integer = 3)

    (model, model_Δ) = @time initModel(model_params, X, y)    
    task = @time initTask(task_params, X, y)
    predictor = @time FMPredictor(task, model, model_Δ)
    best_predictor = copy(predictor)
    best_auc = -Inf

    # Train the predictor using SGD
    kfds = kfolds(1:X.m, k = kₙ)
    for fd in 1:kₙ
        X_train, X_valid, y_train, y_valid = X[kfds[fd][1], :], X[kfds[fd][2], :], y[kfds[fd][1]], X[kfds[fd][2]]
        for epoch in 1:sgd.num_epochs
            #@show "[SGD - Epoch $epoch] Start..."
            cross_terms = X_train * predictor.model.V
            predictions = sigmoid.(-predictor.model.b .- X_train * predictor.model.u .- sum(cross_terms .^ 2 .- X_train.^2 * predictor.model.V.^2, dims = 2) ./ 2)
            total_losses = loss_deriv.(fill(predictor.task, X_train.m), predictions, y_train)
            # batch update
            sgd_update!(sgd, predictor.model, predictor.model_Δ, X_train, total_losses, cross_terms)
            #evaluation
            predictions = sigmoid.(-predictor.model.b .- X_valid * predictor.model.u .- sum(cross_terms .^ 2 .- X_valid.^2 * predictor.model.V.^2, dims = 2) ./ 2)
            evaluation, _, _ = roc_auc(y_valid, predictions)
            if evaluation > best_auc
                best_predictor = copy(predictor)
                best_auc = evaluation
            end
            @show "[SGD - Epoch $epoch] Evaluation: $evaluation"        
            #@show "[SGD - Epoch $epoch] End."
        end
    end

    predictor
end

X = sparse([2.0 1.0 3.0; 1.0 1.0 1.0; 1.0 1.0 1.0; 1.0 1.0 1.0])
y = [1.0; 2.0; 3.0; 4.0]
train(X, y)

