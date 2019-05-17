mutable struct testProblem

    model::AbstractNLPModel
    meta::NLPModelMeta #the NLP meta of the NLP model above

    weigthCalls::Number

    function testProblem(model)
        weigthCalls=0
        new(model,model.meta,weigthCalls)
    end
end

mutable struct tunedOptimizer

    name::String
    nb_hyperparam::Int64
    hyperparam_low_bound::Vector{Float64}
    hyperparam_up_bound::Vector{Float64}
    param_cstr::Vector{Function}
    run::Function #should take as arguments an NLP model and a vector of hyperparameters
                  #should return a tuple (calls,success)

    function tunedOptimizer(run,nb_hyperparam)
        name="noname"
        hyperparam_low_bound=[]
        hyperparam_up_bound=[]
        param_cstr=Vector{Function}()
        new(name,nb_hyperparam,hyperparam_low_bound,hyperparam_up_bound,param_cstr,run)
    end
end

function runtopt(solver::tunedOptimizer,pb::testProblem,x::AbstractVector)
    funcCalls=solver.run(pb.model,x)
    NLPModels.reset!(pb.model)
    return funcCalls
end

struct tuningProblem <: AbstractNLPModel

  meta :: NLPModelMeta
  counters :: Counters
  f :: Function
  c :: Function

  function tuningProblem(solver::tunedOptimizer,Pbs::Vector{testProblem};weights=true,fail_penalty=Inf,x0=[],lvar=[],uvar=[])

      function f(x)
          total_calls = 0
          for pb in Pbs
              funcCalls=runtopt(solver,pb,x)
              if funcCalls<Inf
                  total_calls += funcCalls/(weights ? pb.weigthCalls : 1)
              elseif fail_penalty<Inf
                  total_calls += fail_penalty/(weights ? pb.weigthCalls : 1)
              else
                  return Inf
              end
          end
          return total_calls
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
