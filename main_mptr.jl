using NOMAD

MATLAB_path = "/home/pascpier/Documents/MATLAB"

include("/home/pascpier/Documents/mptr/mptr_nomad.jl")
include("structs.jl")
include("metaoptimization.jl")
include("write_csv.jl")

#function to run mptr on a given problem
function runMPTR(model::T, hyperparameters, maxCalls) where T<:AbstractNLPModel
    (solved,funcCalls,gNorms)=mptr_nomad(model,hyperparameters[1],hyperparameters[2], maxCalls)
    return (solved ? funcCalls : Inf)
end

c(x)=x[2]-x[1]

MPTRstruct=tunedOptimizer(runMPTR,2)
MPTRstruct.name="mptr"
MPTRstruct.param_cstr=[c]
MPTRstruct.hyperparam_low_bound=[eps(Float32),eps(Float64)]
MPTRstruct.hyperparam_up_bound=[1,1]
MPTRstruct.weightCalls=Dict("arwhead" => 106,
                            "cosine" => 569,
                            "dqrtic" => 954,
                            "edensch" => 228,
                            "eg2" => 67,
                            "extrosnb" => 27044,
                            "fletchcr" => 279,
                            "freuroth" => 515,
                            "genhumps" => 9234,
                            "genrose" => 1198,
                            "nondquar" => 1473,
                            "schmvett" => 560,
                            "sinquad" => 15423,
                            "vardim" => 138)

function runNOMAD(model::T) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=0
    function eval(x)
        (obj,cstr)=objcons(model,x)
        if obj<Inf
            return (true,true,[obj,cstr[1]])
        else
            return (false,true,[Inf,Inf])
        end
    end
    result=nomad(eval,param)
    return result.best_feasible
end

n=10

#Pbs = [Vardim]

Pbs = [arwhead(n), cosine(n), dqrtic(n),  edensch(n), eg2(n) ,extrosnb(n), fletchcr(n), freuroth(n) ,genhumps(n), genrose(n), nondquar(n), schmvett(n) , #=sinquad(n),=# vardim(n)]

#pb=Vardim;plot_grad_mptr(pb.cost,[0.01,0.001])

#println(runtopt(MPTRstruct,Genhumps,[1e-1,1e-8]))

metaoptimization(Pbs,MPTRstruct,runNOMAD,MATLAB_path;pre_heuristic=true,maxFactor=7,penaltyFactor=1,admitted_failure=0.3)
