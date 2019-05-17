"""

    metaoptimization(Pbs::Vector{testProblem},solver::tunedOptimizer,bb::blackboxOptimizer,data_path::String;grid::Int=20,maxFactor=Inf,log=false,recompute_min=false,pre_heuristic=true)

Run NOMAD.jl on a unique black box function  that sums the numbers of function calls required by `Solver` to optimize all problems
from `Pbs`. Before then, several preliminary heuristics are launched. First, one is launched on each problem that does not have
a `weigthCalls` in order to approximate it. Then, another heuristic is launched on the global problem to determine a
good initialization point along with a bounding box for the optimization.

# **Optional arguments** :

- `log::Bool`

If true, hyperparameters are distributed on a logarithmic grid during heuristic (false by default)

- `recompute_minmax`

If true, a preliminary heuristic is launched on each problem to approximate their weigthCalls even if they already exist (false by default)

"""

function metaoptimization(Pbs_::Vector{testProblem},solver::tunedOptimizer,runBBoptimizer::Function,data_path::String;
                            grid::Int=20,maxFactor::Number=Inf,log::Bool=true,recompute_min::Bool=false,pre_heuristic::Bool=true,
                            hyperparam_x0=[],hyperparam_lb=[],hyperparam_ub=[],fail_penalty=Inf)

    Pbs=deepcopy(Pbs_)

    @info "start guessing weigthCalls for each problem..."
    for pb in Pbs
        if recompute_min || pb.weigthCalls==0
            tpb_csv = tuningProblem(solver,[pb];weights = false)
            (hyperparam_init, minCalls, guess_low_bound, guess_up_bound) = write_csv(tpb_csv,data_path;grid=grid,log=log)
            pb.weigthCalls=minCalls
            println("weigthCalls = $(pb.weigthCalls)")
        end
    end
    @info "all weigthCalls guessed.\n"

    if pre_heuristic
        @info "beginning prelimary heuristic\n"
        tpb_csv = tuningProblem(solver,Pbs;fail_penalty=fail_penalty)
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

    tpb = tuningProblem(solver,Pbs;fail_penalty=fail_penalty,x0=x0,lvar=lvar,uvar=uvar)

    argmin=runBBoptimizer(tpb)

    println("\nhyperparameters found after optimization : $(argmin)")
    for pb in Pbs
        optCalls = runtopt(solver,pb,argmin)
        println(pb.meta.name * " : $optCalls calls")
    end


end
