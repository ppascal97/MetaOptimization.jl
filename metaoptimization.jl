using CSV

include("latinHypercube.jl")

"""

metaoptimization(Pbs::Vector{AbstractNLPModel},solver::tunedOptimizer,runBBoptimizer::Function;
			                Nlhs::Int=0, logarithm::Bool=false, hyperparam_x0=Vector{Number}(),
			                penalty::Number=0,admitted_failure::Number=0.0,load_dict::String="",
			                sgte_ratio::Float64=0.0,valid_pbs::Vector{T}=[],weights::Bool=true)


Run the direct-search algorithm embedded in `runBBoptimizer` to minimize the average weighted performance index returned by `Solver` after running on all problems
from `Pbs`.

# **Arguments** :

- `Pbs::Vector{AbstractNLPModel}`

The set of training problems used for the parameters optimization. The cost to minimize will be the average weighted performance index
defined by `Solver.run`.

- `solver::tunedOptimizer`

the tunedOptimizer depending on the parameters to optimize.

- `runBBoptimizer::Function`

A *Function* that should take as argument an AbstractNLPModel and return a tuple (argmin,min) with argmin the input found by the direct-search algorithm that minimizes
the cost from this model with respect to its constraints, and min the corresponding objective value. Note that meta.x0, meta.lvar and meta.uvar from the NLP model should be used.
To get the objective function and constraints from the NLP model for a given input, it is suggested to use the following method :

    (f,c)=NLPModels.objcons(tpb::AbstractNLPModel, x::AbstractVector)

f is a *Float* corresponding to the objective value. c is a *Float* equal to the constraint :

    fail_nb/length(Pbs)-admitted_failure

with fail_nb the number of runs that failed for the set of parameters `x`. This quantity should remain negative in the feasible space.

It is discouraged to use the methods `NLPModels.obj(tpb::tuningProblem, x::AbstractVector)` and `NLPModels.cons(tpb::tuningProblem, x::AbstractVector)`
because the admitted failure constraint needs the calculation of the cost to be evaluated so it is a waste of time to evaluate
them separately.


# **Optional arguments** :

- `weights::Bool`

If set to true, the performance indexes are weighted with the values available in `solver.weightPerf` before being incorporated into the objective function.
`true` by default`

- `Nlhs::Int`

If strictly positive, a Latin-Hypercube search will be performed before calling the direct-search algorithm in order to find a good initialization point.
More precisely, n*Nlhs sample points will be tested with n the number of parameters to tune.

- `logarithm::Bool`

If true, the Latin-Hypercube search and the direct-search process will be performed with a logarithmic scale.

- `hyperparam_x0::Vector{Number}`

set an initialization point for the optimization. It will not be used if Nlhs>0.
By default, and if Nlhs==0, the initialization point will be at the center of the bounding box defined in the tunedOptimizer meta (at the
logarithmic center if logarithm is set to true).

- `penalty::Number`

If an optimization fails during the calculation of the global cost to minimize, the weighted performance index of the corresponding
training problem is replaced by `penalty`. If penalty is set to 0, when a problem fails, it is not counted in the average.

- `admitted_failure::Number`

proportion of failures admitted when computing global cost.
0.0 by default.

- `load_dict::String`

If it is a valid existing .csv file name, solver.weightPerf is replaced with the data from this file at the beginning of the process.
empty by default.

- `sgte_ratio::Float`

If strictly positive, a surrogate function of type AbstractNLPModel will be provided to `runBBoptimizer` as an optional second argument (of keyword sgte).
This surrogate approximates the global cost by randomly picking a given number of problems among all training problems and by computing their average
weighted (or not) performance index. `sgte_ratio` correspond of the proportion of training problems that will be picked to constitute the surrogate.

- `valid_pbs::Vector{AbstractNLPModel}`

Set of problems used to validate optimal parameters found by the direct-search algorithm. The average weighted (or not) performance index from validation
problems will be computed and displayed.

"""

function metaoptimization(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function;
                            Nlhs::Int=0, logarithm::Bool=false, hyperparam_x0=Vector{Number}(),
                            penalty::Number=0,admitted_failure::Number=0.0,load_dict::String="",
                            sgte_ratio::Float64=0.0,valid_pbs::Vector{T}=[],weights::Bool=true) where T<:AbstractNLPModel

    sgte_num=Int(round(sgte_ratio*length(Pbs))) #Number of problems used for surrogate
    0<=sgte_num<length(Pbs) || error("wrong surrogate ratio")
    admitted_failure>0 || (penalty=Inf)

	if weights
		#load weights dictionary
	    if !(load_dict=="") #loading weights dictionary
	        try
	            df=CSV.read(load_dict)
	            solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
	            @info "loaded $load_dict (optimizing)"
	        catch
	            @warn "could not load $load_dict"
	        end
	    end

		#check if all weights are available and displaying them
		println("\nweights : ")
		for pb in hcat(Pbs,valid_pbs)
			haskey(solver.weightPerf,pb.meta.name * "_$(pb.meta.nvar)") || error(pb.meta.name * "_$(pb.meta.nvar) is not in weightPerf")
			println(pb.meta.name * "_$(pb.meta.nvar) : $(MPTRstruct.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"])")
		end
		println()
	end

	#generate initialization point
    if Nlhs>0
        @info "beginning lhs preliminary heuristic...\n"
        tpb_lhs = tuningProblem(solver,Pbs;penalty=penalty,admitted_failure=admitted_failure,logarithm=logarithm,weights=weights)
        (hyperparam_init, minPerf) = latinHypercube(tpb_lhs,;N=Nlhs)
        println("initialization with hyperparameters : $hyperparam_init")
		x0 = hyperparam_init
        logarithm && (x0=exp.(x0))
    elseif isempty(hyperparam_x0)
        if logarithm
            x0=exp.((log.(solver.hyperparam_up_bound)+log.(solver.param_low_bound))/2)
        else
            x0=(solver.hyperparam_up_bound+solver.param_low_bound)/2
        end
    else
        x0 = hyperparam_x0
		logarithm && (x0=exp.(x0))
    end

	#launch meta-optimization process
    tpb = tuningProblem(solver,Pbs;x0=x0,penalty=penalty,admitted_failure=admitted_failure,logarithm=logarithm,weights=weights)
    if sgte_num>0
        tpb_sgte=tuningProblem(solver,Pbs;Nsgte=sgte_num,x0=x0,penalty=penalty,admitted_failure=admitted_failure,logarithm=logarithm,weights=weights)
        (argmin,min)=runBBoptimizer(tpb;sgte=tpb_sgte)
    else
        (argmin,min)=runBBoptimizer(tpb)
    end
    logarithm && (argmin=exp.(argmin))

	#display results
    println("\nhyperparameters found after optimization : $(argmin)")
    println("weighted performances :")
    weighted_perfs=Vector{Float64}()
    for pb in Pbs
        perf = runtopt(solver,pb,argmin)
        if perf<Inf
            push!(weighted_perfs,perf/MPTRstruct.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"])
            println(pb.meta.name * "_$(pb.meta.nvar) : $(weighted_perfs[end])")
        else
            println(pb.meta.name * "_$(pb.meta.nvar) : failure")
        end
    end
    var = sum((weighted_perfs.-min).^2)/length(weighted_perfs) #variance
    println("variance : $var\n")

	#validation
    if !isempty(valid_pbs)
        @info "validation..."
        for pb in valid_pbs
	    	perf = runtopt(solver,pb,argmin)
	    	if perf<Inf
	        	push!(weighted_perfs,perf/MPTRstruct.weightPerf[pb.meta.name * "_$(pb.meta.nvar)"])
	        	println(pb.meta.name * "_$(pb.meta.nvar) : $(weighted_perfs[end])")
	    	else
	        	println(pb.meta.name * "_$(pb.meta.nvar) : failure")
	    	end
        end
        tpb_val = tuningProblem(solver,valid_pbs;penalty=penalty,admitted_failure=admitted_failure,weights=weights)
        (valid_obj,c) = objcons(tpb_val,argmin)
        println("objective value for testing set : $min")
        println("objective value for validation set : $valid_obj")
        println("failure constraint in validation set : $(c[end])")
    end

end
