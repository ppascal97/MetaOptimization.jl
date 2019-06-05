using NOMAD

MATLAB_path = "/home/pascpier/Documents/MATLAB"

include("/home/pascpier/Documents/mptr/mptr_nomad.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("write_csv.jl")

#function to run mptr on a given problem
function runMPTR(model::T, hyperparameters, maxCalls) where T<:AbstractNLPModel
    try
        (solved,funcCalls,gNorms)=mptr_nomad(model,hyperparameters[1],hyperparameters[2], maxCalls)
        return (solved ? funcCalls : Inf)
    catch
        return Inf
    end
end

c(x)=x[2]-x[1]

MPTRstruct=tunedOptimizer(runMPTR,2)
MPTRstruct.name="mptr"
MPTRstruct.param_cstr=[c]
MPTRstruct.hyperparam_low_bound=[eps(Float32),eps(Float64)]
MPTRstruct.hyperparam_up_bound=[1,1]
MPTRstruct.weightCalls=Dict("arglina_10" => 67, "arglinb_10" => 57, "arglinc_10" => 57, "bdqrtic_100" => 410,
                            "arwhead_10" => 106, "cosine_10" => 569, "dqrtic_10" => 954, "edensch_10" => 228,
                            "eg2_10" => 67, "extrosnb_10" => 27044, "fletchcr_10" => 279, "freuroth_10" => 515,
                            "genhumps_10" => 9234, "genrose_10" => 1198, "nondquar_10" => 1473, "schmvett_10" => 560,
                            "sinquad_10" => 15423, "vardim_10" => 138)

function runNOMAD(model::T) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=400
    function eval(x)
        (f,c)=objcons(model,x)
        if f<Inf
            return (true, true, [f,c[1],c[2]])
        else
            if x==model.meta.x0
                return (true, false, [Inf,Inf,Inf])
            else
                return (false, false, [Inf,Inf,Inf])
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


#weighting(All_bs,MPTRstruct,runNOMAD,MATLAB_path;recompute_weights=true,bb_computes_weights=true)

#metaoptimization(All_pbs,MPTRstruct,runNOMAD,MATLAB_path; penaltyFactor=0, pre_heuristic=false,admitted_failure=0.2,load_dict="weightCalls.csv")

using CSV
df=CSV.read("weightCalls.csv")
MPTRstruct.weightCalls=Dict(String.(df[:,1]) .=> Int.(df[:,2]))
tpb = tuningProblem(MPTRstruct,All_pbs;penaltyFactor=0,admitted_failure=0.5)
@time println(tpb.f([1e-3,1e-8]))
@time println(tpb.f([1e-3,1e-8]))
