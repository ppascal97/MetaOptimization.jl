using NOMAD

MATLAB_path = "/home/pascpier/Documents/MATLAB"

include("/home/pascpier/Documents/mptr/mptr_nomad.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("write_csv.jl")

#function to run mptr on a given problem
function runMPTR(model::T, hyperparameters) where T<:AbstractNLPModel
    try
        (solved,funcCalls,qual)=mptr_nomad(model,hyperparameters[1],hyperparameters[2], Inf)
        return (funcCalls,qual)
    catch
        return (Inf,0)
    end
end

c(x)=x[2]-x[1]

MPTRstruct=tunedOptimizer(runMPTR,2)
MPTRstruct.name="mptr"
MPTRstruct.param_cstr=[c]
MPTRstruct.hyperparam_low_bound=[eps(Float32),eps(Float64)]
MPTRstruct.hyperparam_up_bound=[1,1]

function runNOMAD(model::T) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=400
    function eval(x)
        (f,c)=objcons(model,x)
        if f<Inf
            return (true, true, [f,c[1]])
        else
            if x==model.meta.x0
                return (true, false, [Inf,Inf])
            else
                return (false, false, [Inf,Inf])
            end
        end
    end
    result=nomad(eval,param)
    return result.best_feasible
end

n=10

Pbs = [vardim(n),cosine(n)]

All_pbs = [arglina(), arglinb(), arglinc(), arwhead(), bdqrtic(), #=beale(),=# broydn7d(), #=brybnd(),=# chainwoo(), chnrosnb_mod(), cosine(), cragglvy(), dixmaane(), dixmaani(), dixmaanm(), dixon3dq(), dqdrtic(), dqrtic(), edensch(), eg2(), engval1(), errinros_mod(), #=extrosnb(),=# fletcbv2(), fletcbv3_mod(), fletchcr(),
                      freuroth(), genhumps(), genrose(), genrose_nash(), indef_mod(), liarwhd(), morebv(), ncb20(), ncb20b(), noncvxu2(), noncvxun(), nondia(), nondquar(), NZF1(), penalty2(), #=penalty3(),=# powellsg(), power(), quartc(), #=sbrybnd(),=# schmvett(), scosine(), sparsine(), sparsqur(), srosenbr(), sinquad(), tointgss(), tquartic(), tridia(), vardim(), woods()]


weighting(Pbs,MPTRstruct,runNOMAD;lhs=false,logarithm=true,recompute_weights=true,save_dict="weightPerf.csv",load_dict="weightPerf.csv")

metaoptimization(Pbs,MPTRstruct,runNOMAD; logarithm=true,lhs=false,load_dict="weightPerf.csv")

#println(mptr_nomad(vardim(n), 0.3301557461, 2.220446049e-16, Inf))

#=using CSV
df=CSV.read("weightPerf.csv")
MPTRstruct.weightPerf=Dict(String.(df[:,1]) .=> Int.(df[:,2]))
tpb = tuningProblem(MPTRstruct,All_pbs;penaltyFactor=0,admitted_failure=0.5)
@time println(tpb.f([1e-3,1e-8]))
@time println(tpb.f([1e-3,1e-8]))=#
