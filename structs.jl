"""

    tunedAlgorithm

mutable struct defining the optimizer to tune. Its performance should depend on a given
number of parameters. Also it should run on NLP models.

# **Constructor** :

    `solver = tunedOptimizer(run::Function,nb_param::Int)`

# **Attributes** :

- `nb_param::Int` :
The number of parameters to tune

-`run::Function` :
A function of the form :

    run(model<:AbstractNLPModel, param::Vector{Float})

with `model` an NLP model and `param` a set of parameters. It should return an index (*Float*) measuring the performance of the run performed
on `model` with parameters `param`. It should return `Inf` if the run failed.

- `name::String` :
A string used as a identifier
`nonamesolver` by default

- `param_low_bound::Vector{Float}` :
A lower bound for parameters
`zeros(Float64,nb_param)` by default.

- `param_up_bound::Vector{Float}` :
A upper bound for parameters
`ones(Float64,nb_param)` by default.

- `param_cstr::Vector{Function}`
A set of functions defining constraints on parameters that should remain negative in the feasible set.
These functions should take as argument a vector of length `nb_param` and return a *Float*.
empty by default.

- `weightPerf::Dict{String,Float}`
A dictionary storing weights used to balance training problems according to their typical performance when computing
the global cost to minimize.
empty by default.

"""
mutable struct tunedOptimizer

    name::String
    nb_param::Int64
    param_low_bound::Vector{Float64}
    param_up_bound::Vector{Float64}
    param_cstr::Vector{Function}
    run::Function
    weightPerf::Dict{String,Float64}

    function tunedOptimizer(run,nb_param)
        name="nonamesolver"
        param_low_bound=zeros(Float64,nb_param)
        param_up_bound=ones(Float64,nb_param)
        param_cstr=Vector{Function}()
        weightPerf=Dict()
        new(name,nb_param,param_low_bound,param_up_bound,param_cstr,run,weightPerf)
    end
end

function runtopt(solver::tunedOptimizer,pb::T,x::AbstractVector) where T<:AbstractNLPModel
    perf=solver.run(pb,x)
    NLPModels.reset!(pb)
    return perf
end

struct tuningProblem <: AbstractNLPModel

  meta :: NLPModelMeta
  counters :: Counters
  f :: Function
  constraint :: Function
  admit_failure::Bool

  function tuningProblem(solver::tunedOptimizer,Pbs::Vector{T};Nsgte=0,weights=true,x0=[],penalty=0,admitted_failure=0.0,logarithm=false) where T<:AbstractNLPModel

      function f(x)

          if Nsgte==0
              index=collect(1:length(Pbs))
          else
              avail=collect(1:length(Pbs))
              index=Vector{Int64}(undef,Nsgte)
              for i=1:Nsgte
                  k=rand(1:length(avail))
                  index[i]=avail[k]
                  deleteat!(avail,k)
              end
          end

          total_perf = 0
          failure=0
          for i in index
              perf = (logarithm ? runtopt(solver,Pbs[i],exp.(x)) : runtopt(solver,Pbs[i],x))
              if perf<Inf
                  if weights
                      w = solver.weightPerf[Pbs[i].meta.name * "_$(Pbs[i].meta.nvar)"]
                      (w == 0) || (total_perf += perf/w)
                  else
                      total_perf += perf
                  end
              elseif penalty==0
                  failure+=1
              elseif penalty<Inf
                  total_perf += penalty
              else
                  return (Inf,Inf)
              end
          end
          return (total_perf/(length(index)-(penalty==0 ? failure : 0)),failure/length(index)-admitted_failure)
      end

      function constraint(x)
          cons = Vector{Float64}()
          for c in solver.param_cstr
              push!(cons,c(x))
          end
          return cons
      end

      name = (length(Pbs)>1 ? solver.name * "_global" : solver.name * "_" * Pbs[1].meta.name * "_$(Pbs[1].meta.nvar)")
      nvar = solver.nb_param
      lvar=(logarithm ? log.(solver.param_low_bound) : solver.param_low_bound)
      uvar=(logarithm ? log.(solver.param_up_bound) : solver.param_up_bound)
      if isempty(x0)
          x0=(uvar+lvar)/2
      else
          x0=(logarithm ? log.(x0) : x0)
      end

      meta = NLPModelMeta(nvar, x0=x0, name=name, lvar=lvar, uvar=uvar)
      counters = Counters()
      new(meta,counters,f,constraint,(admitted_failure>0))
  end

end

function NLPModels.obj(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_obj)
    (obj, cstr) = tpb.f(x)
    return obj
end

function NLPModels.cons(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_cons)
    c = tpb.constraints(x)
    if tpb.admit_failure
        NLPModels.increment!(tpb, :neval_obj)
        (obj, cstr) = tpb.f(x)
        push!(c,cstr)
        return c
    else
        push!(c,-1)
        return c
    end
end

function NLPModels.objcons(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_cons)
    c = tpb.hyperparam_cstr(x)
    for cstr in c
        if cstr>0
            return (Inf, Inf)
        end
    end
    NLPModels.increment!(tpb, :neval_obj)
    (obj, fail_cstr) = tpb.f(x)
    return (obj, fail_cstr)
end
