mutable struct testProblem

    model::AbstractNLPModel
    meta::NLPModelMeta #the NLP meta of the NLP model above

    weightCalls::Number

    function testProblem(model)
        weightCalls=0
        try
            typeof(model.meta)==NLPModelMeta
        catch
            error("NLP model used to construct testProblem needs to have field meta of type NLPModelMeta")
        end
        new(model,model.meta,weightCalls)
    end
end

mutable struct tunedOptimizer

    name::String
    nb_hyperparam::Int64
    hyperparam_low_bound::Vector{Float64}
    hyperparam_up_bound::Vector{Float64}
    param_cstr::Vector{Function}
    run::Function #should take as arguments an NLP model, a vector of hyperparameters and the maximum number of function calls allowed
                  #should return the number of function calls needed (Inf if the optimization failed)

    function tunedOptimizer(run,nb_hyperparam)
        name="nonamesolver"
        hyperparam_low_bound=zeros(Float64,nb_hyperparam)
        hyperparam_up_bound=ones(Float64,nb_hyperparam)
        param_cstr=Vector{Function}()
        new(name,nb_hyperparam,hyperparam_low_bound,hyperparam_up_bound,param_cstr,run)
    end
end

function runtopt(solver::tunedOptimizer,pb::testProblem,x::AbstractVector; maxFactor::Number=Inf)
    funcCalls=solver.run(pb.model,x, maxFactor*pb.weightCalls)
    NLPModels.reset!(pb.model)
    return funcCalls
end

struct tuningProblem <: AbstractNLPModel

  meta :: NLPModelMeta
  counters :: Counters
  f :: Function
  c :: Function

  function tuningProblem(solver::tunedOptimizer,Pbs::Vector{testProblem};weights=true,x0=[],lvar=[],uvar=[],maxFactor=Inf,penaltyFactor=1,admitted_failure=0.0)

      function f(x)
          total_calls = 0
          failure=0
          for pb in Pbs
              funcCalls=runtopt(solver,pb,x;maxFactor=maxFactor)
              if funcCalls<maxFactor*pb.weightCalls
                  total_calls += funcCalls/(weights ? pb.weightCalls : 1)
              elseif weights && admitted_failure>0 && maxFactor<Inf && penaltyFactor<Inf
                  total_calls += penaltyFactor*maxFactor
                  failure+=1
              else
                  return Inf
              end
          end
          if failure/length(Pbs)>admitted_failure
              return Inf
          else
              return total_calls
          end
      end

      function c(x)
          cons = Vector{Float64}()
          for c in solver.param_cstr
              push!(cons,c(x))
          end
          return cons
      end

      name = (length(Pbs)>1 ? solver.name * "_global" : solver.name * "_" * Pbs[1].meta.name * "_$(Pbs[1].meta.nvar)")
      nvar = solver.nb_hyperparam

      if isempty(x0)
          x0=zeros(Float64, nvar)
      end
      if isempty(lvar)
          lvar=solver.hyperparam_low_bound
      end
      if isempty(uvar)
          uvar=solver.hyperparam_up_bound
      end

      meta = NLPModelMeta(nvar, x0=x0, name=name, lvar=lvar, uvar=uvar)
      counters = Counters()
      new(meta,counters,f,c)
  end

end

function NLPModels.obj(tpb::tuningProblem, x::AbstractVector)
  NLPModels.increment!(tpb, :neval_obj)
  return tpb.f(x)
end

function NLPModels.cons(tpb::tuningProblem, x::AbstractVector)
  NLPModels.increment!(tpb, :neval_cons)
  return tpb.c(x)
end

function NLPModels.objcons(tpb::tuningProblem, x::AbstractVector)
    c = cons(tpb,x)
    for cstr in c
        cstr<=0 || return (Inf,nothing)
    end
    f = obj(tpb,x)
    return (f, c)
end
