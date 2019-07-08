using NOMAD, CUTEst

include("/home/pascpier/Documents/NCL/src/NCLSolve.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("benchmark_set.jl")
include("test_problems.jl")

function runNCL(nlp::T,parameters) where T <: AbstractNLPModel
    t = @timed exit = NCLSolve(nlp;
                        scale_penal=parameters[1],
                        scale_tol=parameters[2],
                        scale_constr_viol_tol=parameters[3],
                        scale_compl_inf_tol=parameters[4],
                        acc_factor=parameters[5])
    if exit == 1
        return t[2]
    elseif exit == 2
        return t[2]*parameters[5]*2
    elseif exit == 3
        return Inf
    else
        error("strange exit : $exit")
    end
end

NCLstruct=tunedOptimizer(runNCL,5)
NCLstruct.name="ncl"
NCLstruct.param_low_bound=[1,eps(Float64),eps(Float64),eps(Float64),1]
NCLstruct.param_up_bound=[1000,1,1,1,1000]

sgte_cost=5

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=0
    param.sgte_cost=sgte_cost
    param.seed=-1
    #param.max_bb_eval=2

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

All_Pbs = CUTEst.select(min_con=1, only_free_var=true, only_nonlinear_con=true)

println("selected problems :")
Pbs = Vector{String}()
for i=1:50
    k = rand(1:length(All_Pbs))
    println(All_Pbs[k])
    push!(Pbs,All_Pbs[k])
    deleteat!(All_Pbs,k)
end

weight_dict = "NCLweightPerf"

weighting(Pbs,NCLstruct,runNOMAD;logarithm=true,recompute_weights=false,save_dict=weight_dict,load_dict=weight_dict)

(test_pbs,valid_pbs) = benchmark_set(Pbs,NCLstruct;valid_ratio=0.1,load_dict=weight_dict)

metaoptimization(test_pbs,NCLstruct,runNOMAD; logarithm=true,load_dict=weight_dict,sgte_ratio=(1/sgte_cost),valid_pbs=valid_pbs,admitted_failure=0.08)
