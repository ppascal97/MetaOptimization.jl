using NOMAD, CUTEst

path_to_mptr = "..."

include(path_to_mptr)
include("../structs.jl")
include("../metaoptimization.jl")
include("../weighting.jl")
include("../benchmark_set.jl")
include("../test_problems.jl")

#function to run mptr on a given problem
function runMPTR1(model::T, parameters) where T<:AbstractNLPModel
    try
        (solved,funcCalls,qual,gNorms,Clist)=mptr_nomad(model, parameters[1],parameters[2], 300000)
        return funcCalls/qual
    catch
        return Inf
    end
end

c(x)=x[2]-x[1]

MPTRstruct=tunedOptimizer(runMPTR1,2)
MPTRstruct.name="mptr"
MPTRstruct.param_cstr=[c]
MPTRstruct.param_low_bound=[eps(Float32),eps(Float64)]
MPTRstruct.param_up_bound=[1,1]

sgte_cost=5

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.opportunistic_LH=true
    param.LH_init=200
    param.seed=11653

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

L_pbs = [arglina(100), arglinb(100), arglinc(100), arwhead(1000), bdqrtic(1000), beale(1000), broydn7d(1000), brybnd(1000), chainwoo(1000), chnrosnb_mod(1000), cosine(1000), cragglvy(1000), dixmaane(1000), dixmaani(1000), dixmaanm(1000), dixon3dq(1000),
          dqdrtic(1000), dqrtic(1000), edensch(1000), eg2(1000), engval1(1000), errinros_mod(1000), extrosnb(1000), fletcbv2(1000), fletcbv3_mod(1000), fletchcr(1000), freuroth(1000), genhumps(1000), genrose(1000), genrose_nash(1000), indef_mod(1000),
          liarwhd(1000), morebv(100), ncb20b(500), noncvxu2(1000), noncvxun(1000), nondia(1000), nondquar(1000), NZF1(1000), penalty2(1000), penalty3(1000), powellsg(1000), power(10000), quartc(10000), sbrybnd(1000), schmvett(1000), scosine(1000),
          sparsine(1000), sparsqur(1000), srosenbr(1000), sinquad(1000), tointgss(1000), tquartic(1000), tridia(10000), woods(1000)]


M_pbs = [arglina(10), arglinb(10), arglinc(10), arwhead(100), bdqrtic(100), beale(100), broydn7d(100) ,brybnd(100), chainwoo(100), chnrosnb_mod(100), cosine(100), cragglvy(100), dixmaane(100), dixmaani(100), dixmaanm(100), dixon3dq(100),
          dqdrtic(100), dqrtic(100), edensch(100), eg2(100), engval1(100), errinros_mod(100), extrosnb(100), fletcbv2(100), fletcbv3_mod(100), fletchcr(100), freuroth(100), genhumps(100), genrose(100), genrose_nash(100), indef_mod(100),
          liarwhd(100), morebv(10), ncb20b(50), noncvxu2(100), noncvxun(100), nondia(100), nondquar(100), NZF1(100), penalty2(100) ,penalty3(100), powellsg(100), power(1000), quartc(1000), sbrybnd(100), schmvett(100), scosine(100),
          sparsine(100), sparsqur(100), srosenbr(100), sinquad(100), tointgss(100) ,tquartic(100), tridia(1000), woods(100)]

S_pbs = [arglina(5), arglinb(5), arglinc(5), arwhead(10), bdqrtic(10) ,beale(10), broydn7d(10) ,brybnd(10), chainwoo(10), chnrosnb_mod(10), cosine(10), cragglvy(10), dixmaane(10), dixmaani(10), dixmaanm(10), dixon3dq(10), dqdrtic(10),
                dqrtic(10), edensch(10), eg2(10), engval1(10), errinros_mod(10), extrosnb(10), fletcbv2(10), fletcbv3_mod(10), fletchcr(10), freuroth(10), genhumps(10), genrose(10), genrose_nash(10), indef_mod(10), liarwhd(10), morebv(5),
                ncb20b(20), noncvxu2(10), noncvxun(10), nondia(10), nondquar(10) ,NZF1(10), penalty2(10), penalty3(10), powellsg(10), power(100), quartc(100), sbrybnd(10), schmvett(10), scosine(10), sparsine(10), sparsqur(10), srosenbr(10),
                sinquad(10), tointgss(10) ,tquartic(10), tridia(100), vardim(10), woods(10)]

XS_pbs = [arglina(2), arglinb(2), arglinc(2), arwhead(2), bdqrtic(5), beale(2), broydn7d(2), brybnd(2), chainwoo(2), chnrosnb_mod(2), cosine(2), cragglvy(2), dixmaane(2), dixmaani(2), dixmaanm(2), dixon3dq(2), dqdrtic(2),
                dqrtic(2), edensch(2), eg2(2), engval1(2), errinros_mod(2), extrosnb(2), fletcbv2(2), fletcbv3_mod(2), fletchcr(2), freuroth(2), genhumps(2), genrose(2), genrose_nash(2), indef_mod(3), liarwhd(2), morebv(2),
                noncvxu2(2), noncvxun(2), nondia(2), nondquar(2), NZF1(2), penalty2(3), penalty3(3), powellsg(2), power(10), quartc(2), sbrybnd(2), schmvett(2), scosine(2), srosenbr(2),
                sinquad(2), tointgss(3), tquartic(2), tridia(10), vardim(2), woods(2)]


Pbs= vcat(S_pbs,M_pbs)

weight_dict = "MPTRweightPerfopt.csv"

weighting(Pbs,MPTRstruct,runNOMAD;logarithm=true,recompute_weights=true,save_dict=weight_dict,load_dict=weight_dict)

(test_pbs,valid_pbs) = benchmark_set(Pbs,MPTRstruct;valid_ratio=0.1,load_dict=weight_dict)

metaoptimization(test_pbs,MPTRstruct,runNOMAD; logarithm=true,load_dict=weight_dict,valid_pbs=valid_pbs,admitted_failure=0.08,avg_ratio=0.8)
