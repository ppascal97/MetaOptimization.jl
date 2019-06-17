using NOMAD

MATLAB_path = "/home/pascpier/Documents/MATLAB"

include("/home/pascpier/Documents/mptr/mptr_nomad.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("benchmark_set.jl")

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

sgte_cost=5

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=0
    param.sgte_cost=sgte_cost

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

    if !isnothing(sgte)
        sgte::T where T<:AbstractNLPModel
        function ev_sgte(x)
            (f,c)=objcons(sgte,x)
            if f<Inf
                return (true, true, [f,c[1],c[2]])
            else
                if x==sgte.meta.x0
                    return (true, false, [Inf,Inf,Inf])
                else
                    return (false, false, [Inf,Inf,Inf])
                end
            end
        end
        result=nomad(eval,param,ev_sgte)
    else
        result=nomad(eval,param)
    end

    return (result.best_feasible, result.bbo_best_feasible[1])
end


All_pbs = [arglina(10), arglinb(10), arglinc(10), arwhead(100), bdqrtic(100), #=beale(100),=# broydn7d(100), #=brybnd(100),=# chainwoo(100), chnrosnb_mod(100), cosine(100), cragglvy(100), dixmaane(100), dixmaani(100), dixmaanm(100), dixon3dq(100),
          dqdrtic(100), #=dqrtic(100),=# edensch(100), eg2(100), #=engval1(100),=# errinros_mod(100), #=extrosnb(100),=# fletcbv2(100), fletcbv3_mod(100), fletchcr(100), freuroth(100), genhumps(100), genrose(100), genrose_nash(100), indef_mod(100),
          liarwhd(100), morebv(10), #=ncb20b(50),=# noncvxu2(100), noncvxun(100), nondia(100), nondquar(100), NZF1(100), penalty2(100), #=penalty3(100),=# powellsg(100), power(1000), quartc(1000), #=sbrybnd(100),=# schmvett(100), scosine(100),
          sparsine(100), sparsqur(100), srosenbr(100), sinquad(100), tointgss(100), #=tquartic(100),=# tridia(1000), #=vardim(100),=# woods(100),
          arglina(5), arglinb(5), arglinc(5), arwhead(10), bdqrtic(10), #=beale(10),=# broydn7d(10), #=brybnd(10),=# chainwoo(10), chnrosnb_mod(10), cosine(10), cragglvy(10), dixmaane(10), dixmaani(10), dixmaanm(10), dixon3dq(10), dqdrtic(10),
          dqrtic(10), edensch(10), eg2(10), engval1(10), errinros_mod(10), extrosnb(10), fletcbv2(10), fletcbv3_mod(10), fletchcr(10), freuroth(10), genhumps(10), genrose(10), genrose_nash(10), indef_mod(10), liarwhd(10), morebv(5),
          ncb20b(20), noncvxu2(10), noncvxun(10), nondia(10), nondquar(10), #=NZF1(10),=# penalty2(10), penalty3(10), powellsg(10), power(100), quartc(100), sbrybnd(10), schmvett(10), scosine(10), sparsine(10), sparsqur(10), srosenbr(10),
          sinquad(10), tointgss(10), #=tquartic(10),=# tridia(100), vardim(10), woods(10)]

#All_pbs = [vardim(100)]

#weighting(All_pbs,MPTRstruct,runNOMAD;lhs=false,logarithm=true,recompute_weights=false,save_dict="weightPerf.csv",load_dict="weightPerf.csv")

#(test_pbs,valid_pbs) = benchmark_set(All_pbs;valid_ratio=0.1)

test_pbs = [extrosnb(10) , liarwhd(10) , freuroth(100) , indef_mod(100) , penalty3(10) , freuroth(10) , fletcbv2(10) , genhumps(100) , vardim(10) , scosine(10) , sparsine(100) , indef_mod(10) , woods(100) , power(100) , genrose_nash(100) , fletcbv3_mod(100) , dixon3dq(10) , liarwhd(100) , edensch(10) , eg2(100) , edensch(100) , dixmaani(9) , sbrybnd(10) , noncvxun(10) , arglinc(10) , morebv(5) , dixmaanm(9) , cragglvy(100) , nondquar(10) , powellsg(8) , tointgss(10) , ncb20(20) , dqrtic(10) , penalty2(100) , srosenbr(10) , woods(8) , cragglvy(10) , dqrtic(100) , dixon3dq(100) , powellsg(100) , bdqrtic(100) , chnrosnb_mod(10) , dqrtic(10) , nondia(10) , arglinb(10) , arwhead(10) , tridia(100) , genhumps(10) , chnrosnb_mod(100) , arglinb(5) , schmvett(10) , fletcbv3_mod(10) , errinros_mod(100) , nondia(100) , dixmaani(99) , nondquar(100) , quartc(100) , chainwoo(100) , genrose(10) , dixmaane(99) , fletcbv2(100) , dixmaanm(99) , sparsqur(10) , arglina(5) , quartc(1000) , cosine(10) , fletchcr(10) , dixmaane(9) , arwhead(100) , eg2(10) , bdqrtic(10) , noncvxu2(10) , NZF1(100) , arglinc(5) , genrose_nash(10) , schmvett(100) , noncvxu2(100) , errinros_mod(10) , morebv(10) , tointgss(100) , power(1000) , chainwoo(8) , srosenbr(100) , sinquad(10) , scosine(100) , genrose(100) , penalty2(10) , arglina(10) ]

valid_pbs=[broydn7d(100) , eg2(10) , sinquad(100) , fletchcr(100) , sparsine(10) , cosine(100) , broydn7d(10) , tridia(1000) , sparsqur(100) , noncvxun(100)]


metaoptimization(test_pbs,MPTRstruct,runNOMAD; logarithm=true,lhs=false,load_dict="weightPerf.csv",sgte_ratio=(1/sgte_cost),valid_pbs=valid_pbs,hyperparam_x0=exp.([-0.4018000000, -2.8175000000]),admitted_failure=0.05)

#println(mptr_nomad(arwhead(10), 0.00376548, 6.88008e-12, Inf))

#=using CSV
df=CSV.read("weightPerf.csv")
MPTRstruct.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
tpb = tuningProblem(MPTRstruct,All_pbs;Nsgte=1)
@time println(tpb.f([1e-2,1e-6]))=#
