using Suppressor, NOMAD, CUTEst

include("/home/pascpier/Documents/NCL/src/NCL.jl")
include("structs.jl")
include("weighting.jl")
include("metaoptimization.jl")
include("benchmark_set.jl")
include("test_problems.jl")

function runNCL1(nlp::T,parameters) where T <: AbstractNLPModel
	reset!(nlp.counters)
	exit = try
	    exit = @suppress begin
	        exit = NCL.NCLSolve(nlp;
		                scale_penal=parameters[1],
		                scale_tol=parameters[2],
		                scale_constr_viol_tol=parameters[3],
		                scale_compl_inf_tol=parameters[4],
		                scale_mu_init=parameters[5],
						print_level_NCL=0)
	    end
	    exit
	catch e
	    @warn e
	    println("problem : ", nlp.meta.name," ; parameters : ",parameters)
	    3
	end
	cost = nlp.counters.neval_obj
		+ nlp.counters.neval_cons
		+ nlp.counters.neval_grad
		+ nlp.counters.neval_jac
		+ nlp.counters.neval_hess
	if exit == 1
	    return cost
	elseif exit == 2
	    return cost*2
	elseif exit == 3
	    return Inf
	else
	    @warn "strange exit : $exit"
	    return Inf
	end
end

init_penal = 10.0
init_tol = 0.1
init_constr_viol_tol = 0.1
init_compl_inf_tol = 0.1
init_mu = 0.1

min_tol = 1e-10
min_constr_viol_tol = 1e-10
max_penal = 1e15
min_compl_inf_tol = 1e-8
min_mu = 1e-10

NCLstruct=tunedOptimizer(runNCL1,5)
NCLstruct.name="ncl"
NCLstruct.param_low_bound=[1,min_tol/init_tol,min_constr_viol_tol/init_constr_viol_tol,min_compl_inf_tol/init_compl_inf_tol,min_mu/init_mu]
NCLstruct.param_up_bound=[max_penal/init_penal,1,1,1,1]

sgte_cost=5

function runNOMAD(model::T;sgte=nothing) where T<:AbstractNLPModel
    param = nomadParameters(model.meta.x0,["OBJ","EB"])
    param.lower_bound = model.meta.lvar
    param.upper_bound = model.meta.uvar
    param.LH_init=400
    param.sgte_cost=sgte_cost
    param.seed=-1
	param.opportunistic_LH=true
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

All_Pbs = CUTEst.select(min_con=1, only_nonlinear_con=true)

println("selected problems :")
Pbs = Vector{String}()
for i=1:100
    k = rand(1:length(All_Pbs))
    println(All_Pbs[k])
    push!(Pbs,All_Pbs[k])
    deleteat!(All_Pbs,k)
end

weight_dict = "NCLweightPerf1.csv"

weighting(Pbs,NCLstruct,runNOMAD;logarithm=true,recompute_weights=false,save_dict=weight_dict,load_dict=weight_dict)

(test_pbs,valid_pbs) = benchmark_set(Pbs,NCLstruct;valid_ratio=0.1,load_dict=weight_dict)

metaoptimization(test_pbs,NCLstruct,runNOMAD; logarithm=true,load_dict=weight_dict,sgte_ratio=(1/sgte_cost),valid_pbs=valid_pbs,admitted_failure=0.08)


#=nlp = CUTEstModel("READING9")
parameters =[9.92032e5, 8.02117e-9, 0.000449909, 0.00708341, 1.03587e-15, 1.01959]
exit = NCL.NCLSolve(nlp;
                scale_penal=parameters[1],
                scale_tol=parameters[2],
                scale_constr_viol_tol=parameters[3],
                scale_compl_inf_tol=parameters[4],
                scale_mu_init=parameters[5],
                acc_factor=parameters[6],
		print_level_NCL=1)
cost = nlp.counters.neval_obj
        + nlp.counters.neval_cons
        + nlp.counters.neval_grad
        + nlp.counters.neval_jac
        + nlp.counters.neval_hess
println("cost = $cost")
println("exit = $exit")
dump(nlp.counters)
finalize(nlp)=#
