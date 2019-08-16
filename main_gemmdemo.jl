using Suppressor, NOMAD, NLPModels, Statistics, DataFrames, BenchmarkTools

include("/home/pascpier/Documents/GemmDemo/GemmDemo.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("benchmark_set.jl")
include("test_problems.jl")

mutable struct MATRIXModel <: AbstractNLPModel

    meta::NLPModelMeta
    counters::Counters
    m::Int64
    n::Int64
    k::Int64
    A::Matrix{Float64}
    B::Matrix{Float64}
    Cref::Matrix{Float64}

    function MATRIXModel(m::Int64,n::Int64,k::Int64)
        A = rand(m,k)
        B = rand(k,n)
        C = zeros(m,n)
        Cref = A*B
        meta = NLPModelMeta(m*n)
        counters = Counters()

        #compilation
        gemm_nonpacking!(fill!(C,0), A, B, (m, k, n), Val(12), Val(4))

        new(meta, counters, m, n, k, A, B, Cref)
    end

end

matrixes = Vector{AbstractNLPModel}()
for i=1:20
    m, n, k = 48 .* (i, i, i)
    push!(matrixes,MATRIXModel(m, n, k))
end

function runGemmDemo(nlp::T,parameters;montecarlo=10) where T<:AbstractNLPModel

    cacheM = Int64(parameters[1])
    cacheN = Int64(parameters[2])
    cacheK = Int64(parameters[3])
    cachePsnp = (cacheM, cacheK, cacheN)

    A = nlp.A
    B = nlp.B
    C = nlp.Cref
    Cref = nlp.Cref

    microM, microN = 12, 4

    times = Vector{Float64}()
    try
        for index = 1:montecarlo
            fill!(C,0)
            t = @elapsed gemm_nonpacking!(C, A, B, cachePsnp, Val(microM), Val(microN))
            (C ≈ Cref) || error()
            push!(times,t)
        end
        return minimum(times)
    catch
        return Inf
    end
end

GEMMstruct=tunedOptimizer(runGemmDemo,3)
GEMMstruct.name="gemmdemo"
GEMMstruct.param_low_bound=[12,4,1]

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(["I","I","I"],["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.granularity = [12, 4, 1]
    param.LH_init=400
    param.seed=-1
    #param.max_bb_eval=15

    function eval(x)
        (f,c)=objcons(model,x)
        if f<Inf
            return (true, true, [f,c])
        else
            if x==model.meta.x0
                return (true, false, [Inf,Inf])
            else
                return (false, false, [Inf,Inf])
            end
        end
    end

    result=nomad(eval,param)

    if result.success
        return (result.best_feasible, result.bbo_best_feasible[1])
    else
        return (nothing, Inf)
    end
end

function main()

    M = MATRIXModel(13*48, 13*48, 13*48)

    for i=1:10
        time = runGemmDemo(M,[156,464,55];montecarlo=10)
        println(time)
    end

end

#=function main()

    bestcM = Vector{Number}()
    bestcN = Vector{Number}()
    bestcK = Vector{Number}()
    besttimes = Vector{Number}()

    for matrix in matrixes
        GEMMstruct.param_up_bound=[matrix.m,matrix.n,matrix.k]
        (argmin,min) = metaoptimization([matrix],GEMMstruct,runNOMAD; weights=false, param_x0=[12,4,1])
        push!(bestcM,argmin[1])
        push!(bestcN,argmin[2])
        push!(bestcK,argmin[3])
        push!(besttimes,min)
    end

    println("best cacheM : $bestcM")
    println("best cacheN : $bestcN")
    println("best cacheK : $bestcK")
    println("best run times : $besttimes")

end=#

main()
