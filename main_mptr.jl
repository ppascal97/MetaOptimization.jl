using NOMAD, CUTEst

include("/home/pascpier/Documents/mptr/mptr_nomad.jl")
include("structs.jl")
include("metaoptimization.jl")
include("weighting.jl")
include("benchmark_set.jl")
include("test_problems.jl")

#function to run mptr on a given problem
function runMPTR1(model::T, parameters) where T<:AbstractNLPModel
    try
        (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model,parameters[1],parameters[2], 300000)
        return funcCalls/qual
    catch
        return Inf
    end
end

#function to run mptr on a given problem
function runMPTR2(model::T, parameters) where T<:AbstractNLPModel
    try
        println(parameters)
        println(model.meta.name)
        t = @timed (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model,parameters[1],parameters[2], 300000)
        println(qual)
        return 1000*t[2]/qual
    catch e
        println(e)
        return Inf
    end
end

#function to run mptr on a given problem
function runMPTR3(model::T, parameters) where T<:AbstractNLPModel
    try
        (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model,parameters[1],parameters[2], 300000)
        return funcCalls/(qual^2)
    catch
        return Inf
    end
end

#function to run mptr on a given problem
function runMPTR4(model::T, parameters) where T<:AbstractNLPModel
    try
        t = @timed (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model,parameters[1],parameters[2], 300000)
        return t[2]
    catch
        return Inf
    end
end
0.220184, 0.000406767
#function to run mptr on a given problem
function runMPTR5(model::T, parameters) where T<:AbstractNLPModel
    try
        t = @timed (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model,parameters[1],parameters[2], 50000)
        return 1000*t[2]/qual
    catch
        return Inf
    end
end

c(x)=x[2]-x[1]

MPTRstruct=tunedOptimizer(runMPTR2,2)
MPTRstruct.name="mptr"
MPTRstruct.param_cstr=[c]
MPTRstruct.param_low_bound=[eps(Float32),eps(Float64)]
MPTRstruct.param_up_bound=[1,1]

sgte_cost=5

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=400
    param.sgte_cost=sgte_cost
    param.seed=-1

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

    if !isnothing(sgte)
        sgte::T where T<:AbstractNLPModel
        @info "surrogate is active"
        function ev_sgte(x)
            (f,c)=objcons(sgte,x)
            if f<Inf
                return (true, true, [f,c])
            else
                if x==sgte.meta.x0
                    return (true, false, [Inf,Inf])
                else
                    return (false, false, [Inf,Inf])
                end
            end
        end
        result=nomad(eval,param;surrogate=ev_sgte)
    else
        result=nomad(eval,param)
    end

    if result.success
        return (result.best_feasible, result.bbo_best_feasible[1])
    else
        return (nothing, Inf)
    end
end

Pbs = [arwhead(2),beale(2)]

weight_dict = "MPTRweightPerf2"

weighting(Pbs,MPTRstruct,runNOMAD;logarithm=true,recompute_weights=true,save_dict=weight_dict,load_dict=weight_dict)

(test_pbs,valid_pbs) = benchmark_set(Pbs,MPTRstruct;valid_ratio=0.5,load_dict=weight_dict)

metaoptimization(test_pbs,MPTRstruct,runNOMAD; logarithm=true,load_dict=weight_dict,sgte_ratio=(1/sgte_cost),valid_pbs=valid_pbs,admitted_failure=0.08)

#ε = [0.220184, 0.000406767]

#t = @timed (solved,funcCalls,qual,gNorms,Clist) = mptr_nomad(arwhead(2), ε[1], ε[2], 50000);println(t[2]);println(funcCalls);println(qual);plot_grad_calls(gNorms,Clist)

#=using CSV
df=CSV.read("weightPerf.csv")
MPTRstruct.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
tpb = tuningProblem(MPTRstruct,[dqrtic(100)],admitted_failure=0.05)
@time println(tpb.f(ε))=#
