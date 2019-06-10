include("latinHypercube.jl")

"""

metaoptimization(Pbs::Vector{AbstractNLPModel},solver::tunedOptimizer,runBBoptimizer::Function,data_path::String;
                            grid::Int=20, log::Bool=true,recompute_weights::Bool=false,pre_heuristic::Bool=true,
                            hyperparam_x0::Vector{Number}=[],hyperparam_lb::Vector{Number}=[],
                            hyperparam_ub::Vector{Number}=[],fail_penalty::Number=Inf)

Run NOMAD.jl on a unique black box function  that computes the average number of function calls required by `Solver` to optimize all problems
from `Pbs`. Before then, several preliminary heuristics are launched. First, one is launched on each problem that does not have
a weight specified in `solver.weightPerf` in order to guess it. Then, another heuristic is launched on the global problem to determine a
good initialization point along with a bounding box for the optimization.

# **Arguments** :

- `Pbs::Vector{AbstractNLPModel}`

The set of testProblems used for the hyperparameters optimization. The cost to minimize will be the sum of the numbers of
function calls needed to solve them

- `solver::tunedOptimizer`

the tunedOptimizer depending on the hyperparameters we seek to optimize.

- `runBBoptimizer::Function`

A *Function* that should take as argument an AbstractNLPModel and return the optimal input found by the black box optimizer that minimizes
the cost from this model with respect to its constraints. Note that meta.x0, meta.lvar and meta.uvar from the NLP model should be used.
To get the objective function and constraints from the NLP model, it is suggested to use the following method :

    (f,c)=NLPModels.objcons(tpb::tuningProblem, x::AbstractVector)

f is a *Float* corresponding to the objective value. c is a *Vector{Float}* containing the constraints of the problem. The first components
of this vector corresponds to the constraints specified in `solver::tunedOptimizer`. If the optional argument `admitted_failure` is set to
a strictly positive value, then the last component of `c` is the value of :

    failure/length(Pbs)-admitted_failure

that should remain negative in the feasible space.

It is discouraged to use the methods `NLPModels.obj(tpb::tuningProblem, x::AbstractVector)` and `NLPModels.cons(tpb::tuningProblem, x::AbstractVector)`
for two reasons : First, the admitted failure constraint needs the calculation of the cost to be evaluated so it is a waste of time to evaluate
them separately. Second, the opporunistic constraint strategy set by the optional argument `opportunistic_cstr` cannot be used if the objective
function and constraints are evaluated separately.

- `data_path::String`

The folder path where .csv files generated during heuristics will be stored.

# **Optional arguments** :

- `grid::Int`

Size of the grid used for preliminary heuristics (a given tuningProblem will be evaluated (grid x grid) times).

- `log::Bool`

If true, hyperparameters are distributed on a logarithmic grid during heuristic.
false by default

- `recompute_weights::Bool`

If true, a preliminary heuristic is launched on each problem to approximate their weights even if they are already specified
in `solver.weightPerf`.
false by default

- `pre_heuristic::Bool`

If true, a preliminary heuristic is launched on the global problem to determine good initialization point and bounds.
true by default

- `hyperparam_x0::Vector{Number}`

set an initialization point for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_low_bound from the tunedOptimizer object provided as input.

- `hyperparam_lb::Vector{Number}`

set a lower bound for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_low_bound from the tunedOptimizer object provided as input.

- `hyperparam_ub::Vector{Number}`

set an upper bound for the optimization. It will not be used if pre_heursitic=true. By default, and if pre_heuristic=false, the initialization
point will be the attribute hyperparam_up_bound from the tunedOptimizer object provided as input.

- `penalty::Number`

During the calculation of the global cost that sums all function calls from the different testProblems, if an optimization fails, the product
penalty*maxFactor is added to the sum instead of the weighted number of function calls. This boils down to add penalty*maxCalls
and then to weight it. If penalty is set to 0, when a problem fails, it is not counted in the average.

- `admitted_failure::Number`

proportion of failures admitted when computing global cost. If strictly superior to 0, the last component of `c` outputed by objcons(tpb,x)
is the value of :

    failure/length(Pbs)-admitted_failure

that should remain negative in the feasible space.
0.0 by default.

- `opportunistic_cstr::Bool`

If true, then the constraints solver.hyperparam_cstr are checked before computing the objective value in `(f,c)=NLPModels.objcons(tpb::tuningProblem, x::AbstractVector)`
If one of the constraints is not strictly positive, then the objective function is not computed and outputed to Inf.

- `save_dict::String`

If it is a valid .csv file name, data from solver.weightPerf will be saved in a .csv file with the given name.
`"weightPerf.csv"` by default.

- `load_dict::String`

If it is a valid existing .csv file name, solver.weightPerf is replaced with the data from this file at the beginning of the process.
`""` by default.

"""

function metaoptimization(Pbs::Vector{T},solver::tunedOptimizer,runBBoptimizer::Function;
                            grid::Int=20, lhs::Bool=true,logarithm::Bool=false,
                            hyperparam_x0=Vector{Number}(),hyperparam_lb=Vector{Number}(), hyperparam_ub=Vector{Number}(),
                            penalty::Number=0,admitted_failure::Number=0.0,
                            opportunistic_cstr::Bool=true,load_dict::String="") where T<:AbstractNLPModel

    admitted_failure>0 || (penalty=Inf)

    if !(load_dict=="")
        try
            df=CSV.read(load_dict)
            solver.weightPerf=Dict(String.(df[:,1]) .=> Float64.(df[:,2]))
            @info "loaded $load_dict"
        catch
            @warn "could not load $load_dict"
        end
    end

    for pb in Pbs
        haskey(solver.weightPerf,pb.meta.name * "_$(pb.meta.nvar)") || error(pb.meta.name * "_$(pb.meta.nvar) is not in weightPerf")
    end

    if lhs
        @info "beginning lhs preliminary heuristic...\n"
        tpb_lhs = tuningProblem(solver,Pbs;penalty=penalty,admitted_failure=admitted_failure,logarithm=logarithm)
        (hyperparam_init, minPerf) = latinHypercube(tpb_lhs,;N=grid)
        println("initialization with hyperparameters : $hyperparam_init")
        logarithm && (hyperparam_init=exp.(hyperparam_init))
        x0 = hyperparam_init
    elseif isempty(hyperparam_x0)
        if logarithm
            x0=exp.((log.(solver.hyperparam_up_bound)+log.(solver.hyperparam_low_bound))/2)
        else
            x0=(solver.hyperparam_up_bound+solver.hyperparam_low_bound)/2
        end
    else
        x0 = hyperparam_x0
    end

    lvar = (isempty(hyperparam_lb) ? solver.hyperparam_low_bound : hyperparam_lb)
    uvar = (isempty(hyperparam_ub) ? solver.hyperparam_up_bound : hyperparam_ub)

    tpb = tuningProblem(solver,Pbs;x0=x0,penalty=penalty,admitted_failure=admitted_failure,opportunistic_cstr=opportunistic_cstr,logarithm=logarithm)

    argmin=runBBoptimizer(tpb)

    logarithm && (argmin=exp.(argmin))

    println("\nhyperparameters found after optimization : $(argmin)")
    for pb in Pbs
        (optCalls,qual) = solver.run(pb,argmin)
        if optCalls<Inf
            println(pb.meta.name * " : $optCalls calls ; $(Int(round(qual))) of solution quality")
        else
            println(pb.meta.name * " : failure")
        end
    end

    return argmin

end
