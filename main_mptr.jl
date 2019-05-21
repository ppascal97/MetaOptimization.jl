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
#These bounds will not be used during optimization because preliminary heuristic provides better bounds

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

Arwhead = testProblem(arwhead(n))
Arwhead.weightCalls = 106

Cosine = testProblem(cosine(n))
Cosine.weightCalls = 569

Dqrtic = testProblem(dqrtic(n))
Dqrtic.weightCalls = 954

Edensch = testProblem(edensch(n))
Edensch.weightCalls = 228

Eg2 = testProblem(eg2(n))
Eg2.weightCalls = 67

Extrosnb = testProblem(extrosnb(n))
Extrosnb.weightCalls = 27044

Fletchcr = testProblem(fletchcr(n))
Fletchcr.weightCalls = 279

Freuroth = testProblem(freuroth(n))
Freuroth.weightCalls = 515

Genhumps = testProblem(genhumps(n))
Genhumps.weightCalls = 9234

Genrose = testProblem(genrose(n))
Genrose.weightCalls = 1198

Nondquar = testProblem(nondquar(n))
Nondquar.weightCalls = 1473

Schmvett = testProblem(schmvett(n))
Schmvett.weightCalls = 560

Sinquad = testProblem(sinquad(n))
Sinquad.weightCalls = 15423

Vardim = testProblem(vardim(n))
Vardim.weightCalls = 138

#Pbs = [Vardim]

Pbs = [Arwhead, Cosine, Dqrtic,  Edensch, Eg2 ,Extrosnb, Fletchcr, Freuroth ,Genhumps, Genrose, Nondquar, Schmvett , #=Sinquad,=# Vardim]

#pb=Vardim;plot_grad_mptr(pb.cost,[0.01,0.001])

#println(runtopt(MPTRstruct,Genhumps,[1e-1,1e-8]))

metaoptimization(Pbs,MPTRstruct,runNOMAD,MATLAB_path;pre_heuristic=false,maxFactor=5,penaltyFactor=1,admitted_failure=0.2)
