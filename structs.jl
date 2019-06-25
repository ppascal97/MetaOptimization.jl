mutable struct tunedOptimizer #{T} where T<:Function

    name::String
    nb_hyperparam::Int64
    hyperparam_low_bound::Vector{Float64}
    hyperparam_up_bound::Vector{Float64}
    param_cstr::Vector{Function}
    run::Function #should take as arguments an NLP model, a vector of hyperparameters and the maximum number of function calls allowed
                  #should return the number of function calls needed (Inf if the optimization failed)

                  #fonctions anonymes

    weightPerf::Dict{String,Float64}

    function tunedOptimizer(run,nb_hyperparam)
        name="nonamesolver"
        hyperparam_low_bound=zeros(Float64,nb_hyperparam)
        hyperparam_up_bound=ones(Float64,nb_hyperparam)
        param_cstr=Vector{Function}()
        weightPerf=Dict()
        new(name,nb_hyperparam,hyperparam_low_bound,hyperparam_up_bound,param_cstr,run,weightPerf)
    end
end

function runtopt(solver::tunedOptimizer,pb::T,x::AbstractVector) where T<:AbstractNLPModel
    (funcCalls,qual)=solver.run(pb,x)
    NLPModels.reset!(pb)
    return funcCalls/qual
end

struct tuningProblem <: AbstractNLPModel

  meta :: NLPModelMeta
  counters :: Counters
  f :: Function
  hyperparam_cstr :: Function
  opportunistic_cstr :: Bool
  post_obj_cstr :: Bool

  function tuningProblem(solver::tunedOptimizer,Pbs::Vector{T};Nsgte=0,weights=true,x0=[],penalty=Inf,admitted_failure=0.0,opportunistic_cstr=true,logarithm=false) where T<:AbstractNLPModel

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
              perf= (logarithm ? runtopt(solver,Pbs[i],exp.(x)) : runtopt(solver,Pbs[i],x))
              if perf<Inf
                  total_perf += perf/(weights ? solver.weightPerf[Pbs[i].meta.name * "_$(Pbs[i].meta.nvar)"] : 1)
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

      function hyperparam_cstr(x)
          cons = Vector{Float64}()
          for c in solver.param_cstr
              push!(cons,c(x))
          end
          return cons
      end

      name = (length(Pbs)>1 ? solver.name * "_global" : solver.name * "_" * Pbs[1].meta.name * "_$(Pbs[1].meta.nvar)")
      nvar = solver.nb_hyperparam

      lvar=(logarithm ? log.(solver.hyperparam_low_bound) : solver.hyperparam_low_bound)
      uvar=(logarithm ? log.(solver.hyperparam_up_bound) : solver.hyperparam_up_bound)

      if isempty(x0)
          x0=(uvar+lvar)/2
      else
          x0=(logarithm ? log.(x0) : x0)
      end

      meta = NLPModelMeta(nvar, x0=x0, name=name, lvar=lvar, uvar=uvar)
      counters = Counters()
      new(meta,counters,f,hyperparam_cstr, opportunistic_cstr, (admitted_failure>0))
  end

end

function NLPModels.obj(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_obj)
    (obj, cstr) = tpb.f(x)
    return obj
end

function NLPModels.cons(tpb::tuningProblem, x::AbstractVector)
    NLPModels.increment!(tpb, :neval_cons)
    c = tpb.hyperparam_cstr(x)
    if tpb.post_obj_cstr
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
    if tpb.opportunistic_cstr
        for cstr in c
            if cstr>0
                push!(c,Inf)
                return (Inf, c)
            end
        end
    end
    NLPModels.increment!(tpb, :neval_obj)
    (obj, fail_cstr) = tpb.f(x)
    if tpb.post_obj_cstr
        push!(c,fail_cstr)
    else
        push!(c,-1)
    end
    return (obj, c)
end
