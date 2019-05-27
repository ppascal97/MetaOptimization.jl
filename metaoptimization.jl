"""

metaoptimization(Pbs::Vector{AbstractNLPModel},solver::tunedOptimizer,runBBoptimizer::Function,data_path::String;
                            grid::Int=20, log::Bool=true,recompute_weights::Bool=false,pre_heuristic::Bool=true,
                            hyperparam_x0::Vector{Number}=[],hyperparam_lb::Vector{Number}=[],
                            hyperparam_ub::Vector{Number}=[],fail_penalty::Number=Inf)

Run NOMAD.jl on a unique black box function  that sums the numbers of function calls required by `Solver` to optimize all problems
from `Pbs`. Before then, several preliminary heuristics are launched. First, one is launched on each problem that does not have
a `weightCalls` in order to approximate it. Then, another heuristic is launched on the global problem to determine a
good initialization point along with a bounding box for the optimization.

# **Arguments** :

- `Pbs::Vector{AbstractNLPModel}`

The set of testProblems used for the hyperparameters optimization. The cost to minimize will be the sum of the numbers of
function calls needed to solve them

- `solver::tunedOptimizer`

the tunedOptimizer depending on the hyperparameters we seek to optimize.

- `runBBoptimizer::Function`

A Function taking as argument an AbstractNLPModel and returning the optimal input found by the black box optimizer that minimizes
the cost of this AbstractNLPModel. Note that meta.x0, meta.lvar and meta.uvar from the NLP model should be used.

- `data_path::String`

The folder path where .csv files generated during heuristics will be stored.

# **Optional arguments** :

- `grid::Int`

Size of the grid used for preliminary heuristics (a given tuningProblem will be evaluated (grid x grid) times).

- `log::Bool`

If true, hyperparameters are distributed on a logarithmic grid during heuristic (false by default).

- `recompute_weights::Bool`

If true, a preliminary heuristic is launched on each problem to approximate their weightCalls even if they already exist (false by default).

- `pre_heuristic::Bool`

If true, a preliminary heuristic is launched on the global problem to determine good initialization point and bounds (true by default).

- `hyperparam_x0::Vector{Number}`

set an initialization point for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_low_bound from the tunedOptimizer object provided as input.

- `hyperparam_lb::Vector{Number}`

set a lower bound for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_low_bound from the tunedOptimizer object provided as input.

- `hyperparam_ub::Vector{Number}`

set an upper bound for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_up_bound from the tunedOptimizer object provided as input.

- `maxFactor::Number`

A NLP model optimization will be considered as failed if the number of function calls exceed maxFactor*weightCalls.

- `penaltyFactor::Number`

During the calculation of the global cost that sums all function calls from the different testProblems, if an optimization fails, the product
penaltyFactor*maxFactor is added to the sum instead of the weighted number of function calls. This boils down to add penaltyFactor*maxCalls
and then to weight it.

- `admitted_failure::Number`

proportion of failures admitted when computing global cost. If more than this proportion of problems fail, the sum of function calls is set
to Inf.

"""

function metaoptimization(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function,data_path::String;
                            grid::Int=20, log::Bool=true,recompute_weights::Bool=false,pre_heuristic::Bool=true,
                            hyperparam_x0=Vector{Number}(),hyperparam_lb=Vector{Number}(), hyperparam_ub=Vector{Number}(),
                            maxFactor::Number=Inf, penaltyFactor::Number=1,admitted_failure::Number=0.1) where T<:AbstractNLPModel

    min_guess = false
    for pb in Pbs
        if recompute_weights || !(haskey(solver.weightCalls,pb.meta.name))
            if !min_guess
                @info "start guessing weightCalls for each problem..."
                min_guess=true
            end
            tpb_csv = tuningProblem(solver,[pb];weights=false)
            (hyperparam_init, minCalls, guess_low_bound, guess_up_bound) = write_csv(tpb_csv,data_path;grid=grid,log=log)
            solver.weightCalls[pb.meta.name]=minCalls
            println("weightCalls = $(solver.weightCalls[pb.meta.name])")
        end
    end
    min_guess ? (@info "all weightCalls guessed.\n") : (@info "all weightCalls already available.\n")

    if pre_heuristic
        @info "beginning prelimary heuristic...\n"
        tpb_csv = tuningProblem(solver,Pbs;maxFactor=maxFactor,penaltyFactor=penaltyFactor,admitted_failure=admitted_failure)
        (hyperparam_init, minCalls, guess_low_bound, guess_up_bound) = write_csv(tpb_csv,data_path;grid=grid,log=log)
        println("initialization with hyperparameters : $hyperparam_init")
        println("low bound used : $guess_low_bound")
        println("up bound used : $guess_up_bound")
        x0 = hyperparam_init
        lvar = guess_low_bound
        uvar = guess_up_bound
    else
        x0 = (isempty(hyperparam_x0) ? solver.hyperparam_low_bound : hyperparam_x0)
        lvar = (isempty(hyperparam_lb) ? solver.hyperparam_low_bound : hyperparam_lb)
        uvar = (isempty(hyperparam_ub) ? solver.hyperparam_up_bound : hyperparam_ub)
    end

    tpb = tuningProblem(solver,Pbs;x0=x0,lvar=lvar,uvar=uvar,maxFactor=maxFactor,penaltyFactor=penaltyFactor,admitted_failure=admitted_failure)

    argmin=runBBoptimizer(tpb)

    println("\nhyperparameters found after optimization : $(argmin)")
    for pb in Pbs
        optCalls = runtopt(solver,pb,argmin)
        if optCalls<maxFactor*solver.weightCalls[pb.meta.name]
            println(pb.meta.name * " : $optCalls calls")
        else
            println(pb.meta.name * " : failure (maxCalls = $(maxFactor*solver.weightCalls[pb.meta.name]))")
        end
    end


end
