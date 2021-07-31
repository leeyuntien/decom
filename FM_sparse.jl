using SparseArrays
using Statistics

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

loss(::ClassificationTask, p::Number, y::Number)       = nlogsig(p, y)
loss_deriv(::ClassificationTask, p::Number, y::Number) = nlogsig_deriv(p, y)

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
    k0::Bool
    k1::Bool
    num_factors::Int64
    mean::Float64
    stddev::Float64
    GaussianModelParams(; k0 = true, k1 = true, num_factors = 8, mean = .0, stddev = .01) =
        new(k0, k1, num_factors, mean, stddev)
end
const gauss = GaussianModelParams

mutable struct FMModel
    k0::Bool
    k1::Bool
    w0::Float64
    w::Vector{Float64}
    V::Matrix{Float64}
    num_factors::Int64
end

struct FMPredictor{T<:PredictorTask}
    task::T
    model::FMModel
end

struct SGDMethod <: MethodParams
    alpha::Float64
    num_epochs::Int64
    # regularization
    reg0::Float64
    regw::Float64
    regv::Float64
    SGDMethod(; alpha::Float64 = 0.01, num_epochs::Int64 = 100, reg0::Float64 = .0, regw::Float64 = .0, regv::Float64 = .0) =
        new(alpha, num_epochs, reg0, regw, regv)
end
const sgd = SGDMethod

function predict_instance!(model::FMModel,
    idx::StridedVector{Int64}, x::StridedVector{Float64},
    f_sum::Vector{Float64}, sum_sqr::Vector{Float64})
    fill!(f_sum, .0)
    fill!(sum_sqr, .0)
    result = zero(Float64)
    if model.k0
        result += model.w0
    end
    if model.k1
        for i in 1:length(idx)
            result += model.w[idx[i]] * x[i]
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

"""Instance prediction specialized for classification"""
function predict_instance!(predictor::FMPredictor,
                           idx::StridedVector{Int64}, x::StridedVector{Float64},
                           f_sum::Vector{Float64}, sum_sqr::Vector{Float64})
    p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
    sigmoid(p)
end

"""Instance prediction specialized for regression"""
function predict_instance!(predictor::FMPredictor,
                           idx::StridedVector{Int64}, x::StridedVector{Float64},
                           f_sum::Vector{Float64}, sum_sqr::Vector{Float64})
    p = predict_instance!(predictor.model, idx, x, f_sum, sum_sqr)
    max(min(p, predictor.task.target_max), predictor.task.target_min)
end

"""Predicts labels for each column of `X` and stores the results into `result`"""
function predict!(predictor::FMPredictor, X::SparseMatrixCSC, result::Vector{Float64})
    fill!(result, .0)
    f_sum = fill(.0, predictor.model.num_factors)
    sum_sqr = fill(.0, predictor.model.num_factors)
    Threads.@threads for c in 1:X.n
        X_nzrange = nzrange(X, c)
        idx = X.rowval[X_nzrange] # sub(X.rowval, X_nzrange)
        x = X.nzval[X_nzrange] # sub(X.nzval, X_nzrange)
        result[c] = predict_instance!(predictor, idx, x, f_sum, sum_sqr)
    end
end

function evaluate!(evaluator::SquaredErrorEvaluator, predictor::FMPredictor,
    X::SparseMatrixCSC, y::Vector{Float64}, predictions::Vector{Float64})
    predict!(predictor, X, predictions)
    err = [sqerr(predictions[i], y[i]) for i in 1:length(y)]
    evaluator.stat(err .* err)
end

function sgd_update!(
        sgd::SGDMethod, model::FMModel, alpha::Float64,
        idx::StridedVector{Int64}, x::StridedVector{Float64},
        mult::Float64, f_sum::Vector{Float64})

    if model.k0
        model.w0 -= alpha * (mult + sgd.reg0 * model.w0)
    end
    if model.k1
        for i in 1:length(idx)
            model.w[idx[i]] -= alpha * (mult * x[i] + sgd.regw * model.w[idx[i]])
        end
    end
    @inbounds for f in 1:model.num_factors
       @inbounds for i in 1:length(idx)
            grad = f_sum[f] * x[i] - model.V[f,idx[i]] * x[i] * x[i]
            model.V[f,idx[i]] -= alpha * (mult * grad + sgd.regv * model.V[f,idx[i]])
        end
    end
end

function sgd_epoch!(
    sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor,
    X::SparseMatrixCSC, y::StridedVector{Float64}, epoch::Int64, alpha::Float64,
    predictions::Vector{Float64}, f_sum::Vector{Float64}, sum_sqr::Vector{Float64})

    p = zero(Float64)
    mult = zero(Float64)

    for c in 1:X.n # X.n = size(y)[1] = number of data points
        X_nzrange = nzrange(X, c)
        idx = X.rowval[X_nzrange] # sub(X.rowval, X_nzrange)
        x = X.nzval[X_nzrange] # sub(X.nzval, X_nzrange)
        #@show "DEBUG: processing $c"
        p = predict_instance!(predictor, idx, x, f_sum, sum_sqr)
        #@show "DEBUG: prediction - p: $p, f_sum: $f_sum, sum_sqr: $sum_sqr"
        mult = loss_deriv(predictor.task, p, y[c])
        predictions[c] = p
        #@show "DEBUG: mult: $mult"
        sgd_update!(sgd, predictor.model, alpha, idx, x, mult, f_sum)
    end
    #evaluation
    # @time evaluation = evaluate!(evaluator, predictor, X, y, predictions)
    err = [sqerr(predictions[i], y[i]) for i in 1:length(y)]
    evaluation = evaluator.stat(err .* err)
    @show "[SGD - Epoch $epoch] Evaluation: $evaluation"
end

function sgd_train!(
    sgd::SGDMethod, evaluator::Evaluator, predictor::FMPredictor,
    X::SparseMatrixCSC, y::StridedVector{Float64})

    predictions = zeros(length(y))
    f_sum = zeros(predictor.model.num_factors)
    sum_sqr = zeros(predictor.model.num_factors)

    @show "Learning Factorization Machines with gradient descent..."
    for epoch in 1:sgd.num_epochs
        #@show "[SGD - Epoch $epoch] Start..."
        @time sgd_epoch!(sgd, evaluator, predictor,
                         X, y, epoch, sgd.alpha,
                         predictions, f_sum, sum_sqr)
        #@show "[SGD - Epoch $epoch] End."
    end
end

function initModel(params::GaussianModelParams, X::SparseMatrixCSC, y::Vector{Float64})
    # initialization
    num_attributes, num_samples = size(X)
    # sanity check
    @assert length(y) == num_samples

    # create initial model
    w0 = .0
    w = zeros(num_attributes)
    V = randn(params.num_factors, num_attributes) .* params.stddev .+ params.mean

    # new model
    model = FMModel(params.k0, params.k1, w0, w, V, params.num_factors)
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

function train(X::SparseMatrixCSC, y::Vector{Float64};
    method::SGDMethod         = sgd(alpha = 0.01, num_epochs = 100, reg0 = .0, regv = .0, regw = .0),
    evaluator::Evaluator      = rmse(),
    task_params::TaskParams   = regression(),
    model_params::ModelParams = gauss(k0 = true, k1 = true, num_factors = 8, mean = .0, stddev = .01))

    model = @time initModel(model_params, X, y)
    task = @time initTask(task_params, X, y)
    predictor = @time FMPredictor(task, model)

    # Train the predictor using SGD
    sgd_train!(method, evaluator, predictor, X, y)

    predictor
end

T = [
5           1 0     1 0 0 0    1 0 0        12.5;
5           1 0     0 1 0 0    1 0 0        20;
4           1 0     0 0 1 0    1 0 0        78;
1           0 1     1 0 0 0    0 0 1        12.5;
1           0 1     0 1 0 0    0 0 1        20;
]

X = sparse(float.(rand(Bool, 500, 10000000)))#sparse(T[:,2:end]')
y = float.(rand(Bool, 10000000) .* 2 .- 1)#T[:,1]

fm = train(X, y)
@show fm
