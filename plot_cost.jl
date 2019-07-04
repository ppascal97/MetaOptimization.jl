using CSV, DataFrames, ProgressMeter

"""

    plot_cost(tpb::tuningProblem,data_path::String;N::Int=20,logarithm::Bool=false)

-> Run the optimizer `solver` on the list of problems `Pbs` for various sets of parameters, distributed on a NxN grid.

-> export the tested parameters along with the corresponding average weighted performance indexes.

-> return useful information about the executed heuristic.

# **Outputs** :

- `best_param::Vector{Float64}`

The couple of hyperparameters that returned the minimal number of function calls.

- `minPerf::Float64`

The minimal cost found during preliminary heuristic

- `guess_low_bound::Vector{Float64}`

Bounds for parameters outside of which all optimization processes failed (lower bound)

- `guess_up_bound::Vector{Float64}`

Same (upper bound)

# **Optional arguments** :

- `N::Int`

Size of the grid used for the heuristic (the tuningProblem will be evaluated (N x N) times).
20 by default.

- `logarithm::Bool`

If true, parameters are distributed on a logarithmic grid.
false by default.

"""
function plot_cost(tpb::tuningProblem,data_path::String;N::Int=20,log::Bool=false)

    tpb.meta.nvar<=2 ? nothing : error("No more than 2 hyperparameters for plotting")

    LB = tpb.meta.lvar
    UB = tpb.meta.uvar

    if tpb.meta.nvar==2

        if logarithm
            if LB[1]==0 || LB[2]==0
                error("log scale needs strict positive lower bounds for hyperparameters")
            end
            X = Vector{Float64}(undef,N)
            X[1] = LB[1]
            rx = (UB[1]/LB[1])^(1/(N-1))
            Y = Vector{Float64}(undef,N)
            Y[1] = LB[2]
            ry = (UB[2]/LB[2])^(1/(N-1))
            for i=2:N
                X[i] = X[i-1] * rx
                Y[i] = Y[i-1] * ry
            end
        else
            X = collect(0:(N-1))*(UB[1]-LB[1])/(N-1) + fill(LB[1],N)
            Y = collect(0:(N-1))*(UB[2]-LB[2])/(N-1) + fill(LB[2],N)
        end

        p = Progress(N^2, 0.1,"heuristic running... ")

        minPerf=Inf
        best_param = Vector{Float64}(undef,tpb.meta.nvar)
        guess_low_bound = []
        guess_up_bound = []
        dfX = DataFrame(A = X)
        dfY = DataFrame(A = Y)
        dfZ = DataFrame()
        for i=1:N
            column = Vector{Float64}(undef,N)
            for j=1:N

                (obj,cstr)=objcons(tpb,[X[i],Y[j]])

                if tpb.admit_failure && cstr>0
                    obj=Inf
                end

                column[j]=obj

                if obj<minPerf
                    minPerf=obj
                    best_param=[X[i],Y[j]]
                end

                if obj<Inf
                    if isempty(guess_low_bound) || isempty(guess_up_bound)
                        guess_low_bound = [X[i],Y[j]]
                        guess_up_bound = [X[i],Y[j]]
                    else
                        if X[i]<guess_low_bound[1]
                            guess_low_bound[1]=X[i]
                        elseif X[i]>guess_up_bound[1]
                            guess_up_bound[1]=X[i]
                        end
                        if Y[j]<guess_low_bound[2]
                            guess_low_bound[2]=Y[j]
                        elseif Y[j]>guess_up_bound[2]
                            guess_up_bound[2]=Y[j]
                        end
                    end
                end
                next!(p)
            end
            dfZ[i]=column
        end

        CSV.write(joinpath(data_path,"X" * tpb.meta.name * ".csv"),  dfX, writeheader=false)
        CSV.write(joinpath(data_path,"Y" * tpb.meta.name * ".csv"),  dfY, writeheader=false)
        CSV.write(joinpath(data_path,"Z" * tpb.meta.name * ".csv"),  dfZ, writeheader=false)

        return (best_param,minPerf,guess_low_bound,guess_up_bound)

    else

        if logarithm
            if LB[1]==0
                error("log scale needs strict positive lower bounds for hyperparameters")
            end
            x = Vector{Float64}(undef,N)
            x[1] = LB[1]
            r = (UB[1]/LB[1])^(1/(N-1))
            for i=2:N
                x[i] = x[i-1] * r
            end
        else
            x = collect(0:(N-1))*(UB[1]-LB[1])/(N-1) + fill(LB[1],N)
        end

        p = Progress(N, 0.1,"$(tpb.meta.name) running... ")

        minPerf=Inf
        best_param = Vector{Float64}(undef,1)
        guess_low_bound = []
        guess_up_bound = []
        dfx = DataFrame(A = x)
        f_x = Vector{Float64}(undef,N)
        for i=1:N

            (obj,cstr)=tpb.output([X[i],Y[j]])

            if tpb.admit_failure && cstr>0
                obj=Inf
            end

            f_x[i]=obj

            if f_x[i]<minPerf
                minPerf=f_x[i]
                best_param=[x[i]]
            end

            if f_x[i]<Inf
                if isempty(guess_low_bound) || isempty(guess_up_bound)
                    guess_low_bound = [x[i]]
                    guess_up_bound = [x[i]]
                else
                    if x[i]<guess_low_bound[1]
                        guess_low_bound[1]=x[i]
                    elseif x[i]>guess_up_bound[1]
                        guess_up_bound[1]=x[i]
                    end
                end
            end
            next!(p)
        end

        dfF=DataFrame(A = f_x)

        CSV.write(joinpath(data_path,"x1D" * tpb.meta.name * ".csv"),  dfx, writeheader=false)
        CSV.write(joinpath(data_path,"F" * tpb.meta.name * ".csv"),  dfF, writeheader=false)

        return (best_param,minPerf,guess_low_bound,guess_up_bound)

    end

end
