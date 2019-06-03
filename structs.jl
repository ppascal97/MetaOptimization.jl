mutable struct tunedOptimizer #{T} where T<:Function

    name::String
    nb_hyperparam::Int64
    hyperparam_low_bound::Vector{Float64}
    hyperparam_up_bound::Vector{Float64}
    param_cstr::Vector{Function}
    run::Function #should take as arguments an NLP model, a vector of hyperparameters and the maximum number of function calls allowed
                  #should return the number of function calls needed (Inf if the optimization failed)

                  #fonctions anonymes

    weightCalls::Dict{String,Int}

    function tunedOptimizer(run,nb_hyperparam)
        name="nonamesolver"
        hyperparam_low_bound=zeros(Float64,nb_hyperparam)
        hyperparam_up_bound=ones(Float64,nb_hyperparam)
        param_cstr=Vector{Function}()
        weightCalls=Dict()
        new(name,nb_hyperparam,hyperparam_low_bound,hyperparam_up_bound,param_cstr,run,weightCalls)
    end
end

function runtopt(solver::tunedOptimizer,pb::T,x::AbstractVector; maxFactor::Number=Inf) where T<:AbstractNLPModel
    if maxFactor<Inf
        funcCalls=solver.run(pb,x, maxFactor*solver.weightCalls[pb.meta.name * "_$(pb.meta.nvar)"])
    else
        funcCalls=solver.run(pb,x, Inf)
    end
    NLPModels.reset!(pb)
    return funcCalls
end

struct tuningProblem <: AbstractNLPModel

  meta :: NLPModelMeta
  counters :: Counters
  calls_and_fail :: Function
  hyperparam_cstr :: Function
  opportunistic_cstr :: Bool
  admit_failure :: Bool

  function tuningProblem(solver::tunedOptimizer,Pbs::Vector{T};weights=true,x0=[],lvar=[],uvar=[],maxFactor=Inf,penaltyFactor=1,admitted_failure=0.0,opportunistic_cstr=true) where T<:AbstractNLPModel

      function calls_and_fail(x)
          total_calls = 0
          failure=0
          for pb in Pbs
              funcCalls=runtopt(solver,pb,x;maxFactor=maxFactor)
              if (!weights && funcCalls<Inf) || (weights && funcCalls<maxFactor*solver.weightCalls[pb.meta.name * "_$(pb.meta.nvar)"])
                  total_calls += funcCalls/(weights ? solver.weightCalls[pb.meta.name * "_$(pb.meta.nvar)"] : 1)
              elseif weights && maxFactor<Inf && penaltyFactor<Inf
                  if penaltyFactor>=1
                      total_calls += penaltyFactor*maxFactor
                  end
                  failure+=1
              else
                  return (Inf,Inf)
              end
          end
          return (total_calls/(length(Pbs)-(penaltyFactor>=1 ? 0 : failure)),failure/length(Pbs)-admitted_failure)
      end

      function hyperparam_cstr(x)
          cons = Vector{Float64}()
          for c in solver.param_cstr
              push!(cons,c(x))
          end
          return cons
      end

      name = (length(Pbs)>1 ? solver.name * "_global" : solver.name * "_" * Pbs[1].meta.name * "_$(Pbs[1].meta.nvar)")
      nvar = solver.nb_hyperparam

      if isempty(x0)
          x0=(solver.hyperparam_up_bound+solver.hyperparam_low_bound)/2
      end
      if isempty(lvar)
          lvar=solver.hyperparam_low_bound
      end
      if isempty(uvar)
          uvar=solver.hyperparam_up_bound
      end

      meta = NLPModelMeta(nvar, x0=x0, name=name, lvar=lvar, uvar=uvar)
      counters = Counters()
      new(meta,counters,calls_and_fail,hyperparam_cstr, opportunistic_cstr, (admitted_failure>0))
  end

end

function NLPModels.obj(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_obj)
    (obj, fail_cstr) = tpb.calls_and_fail(x)
    return obj
end

function NLPModels.cons(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_cons)
    c = tpb.hyperparam_cstr(x)
    if tpb.admit_failure
        NLPModels.increment!(tpb, :neval_obj)
        (obj, fail_cstr) = tpb.calls_and_fail(x)
        push!(c,fail_cstr)
        return c
    else
        return c
    end
end

function NLPModels.objcons(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_cons)
    c = tpb.hyperparam_cstr(x)
    if tpb.opportunistic_cstr
        for cstr in c
            cstr<=0 || return (Inf, (tpb.admit_failure ? push!(c,Inf) : c))
        end
    end
    NLPModels.increment!(tpb, :neval_obj)
    (obj, fail_cstr) = tpb.calls_and_fail(x)
    tpb.admit_failure && push!(c,fail_cstr)
    return (obj, c)
end
