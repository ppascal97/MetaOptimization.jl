using CSV, DataFrames, ProgressMeter

"""

    write_csv(tpb::tuningProblem,data_path::String;grid::Int=20,log::Bool=false)

-> Run the optimizer `solver` on the list of problems `Pbs` for various sets of hyperparameters, distributed on a gridxgrid grid.

-> export the tested hyperparameters along with the corresponding sumed function calls number.

-> return useful information about the executed heuristic.

During heuristic, for a given set of hyperparameters, all problems from `Pbs` will be run and the returned function calls numbers will be sumed.
Note that function calls numbers are inversely weighted with the minimal number of function calls of the given problem (saved in
dictionary `solver.weightCalls`) before being sumed.

# **Outputs** :

- `best_hyperparam::Vector{Float64}`

The couple of hyperparameters that returned the minimal number of function calls.

- `minCalls::Int`

The minimal number of function calls found during preliminary heuristic

- `guess_low_bound::Vector{Float64}`

Bounds for hyperparameters outside of which all optimization processes failed (lower bound)

- `guess_up_bound::Vector{Float64}`

Same (upper bound)

# **Optional arguments** :

- `grid::Bool`

Size of the grid used for the heuristic (the tuningProblem will be evaluated (grid x grid) times).

- `log::Bool`

If true, hyperparameters are distributed on a logarithmic grid

"""
function write_csv(tpb::tuningProblem,data_path::String;grid::Int=20,log::Bool=false)

    tpb.meta.nvar<=2 ? nothing : error("No more than 2 hyperparameters for preliminary heuristic")

    LB = tpb.meta.lvar
    UB = tpb.meta.uvar

    if tpb.meta.nvar==2

        if log
            if LB[1]==0 || LB[2]==0
                error("log scale needs strict positive lower bounds for hyperparameters")
            end
            X = Vector{Float64}(undef,grid)
            X[1] = LB[1]
            rx = (UB[1]/LB[1])^(1/(grid-1))
            Y = Vector{Float64}(undef,grid)
            Y[1] = LB[2]
            ry = (UB[2]/LB[2])^(1/(grid-1))
            for i=2:grid
                X[i] = X[i-1] * rx
                Y[i] = Y[i-1] * ry
            end
        else
            X = collect(0:(grid-1))*(UB[1]-LB[1])/(grid-1) + fill(LB[1],grid)
            Y = collect(0:(grid-1))*(UB[2]-LB[2])/(grid-1) + fill(LB[2],grid)
        end

        p = Progress(grid^2, 0.1,"heuristic running... ")

        minCalls=Inf
        best_hyperparam = Vector{Float64}(undef,tpb.meta.nvar)
        guess_low_bound = []
        guess_up_bound = []
        dfX = DataFrame(A = X)
        dfY = DataFrame(A = Y)
        dfZ = DataFrame()
        for i=1:grid
            column = Vector{Float64}(undef,grid)
            for j=1:grid

                (obj,cstr)=objcons(tpb,[X[i],Y[j]])

                if tpb.admit_failure && cstr[end]>0
                    obj=Inf
                end

                column[j]=obj

                if obj<minCalls
                    minCalls=obj
                    best_hyperparam=[X[i],Y[j]]
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

        return (best_hyperparam,minCalls,guess_low_bound,guess_up_bound)

    else

        if log
            if LB[1]==0
                error("log scale needs strict positive lower bounds for hyperparameters")
            end
            x = Vector{Float64}(undef,grid)
            x[1] = LB[1]
            r = (UB[1]/LB[1])^(1/(grid-1))
            for i=2:grid
                x[i] = x[i-1] * r
            end
        else
            x = collect(0:(grid-1))*(UB[1]-LB[1])/(grid-1) + fill(LB[1],grid)
        end

        p = Progress(grid, 0.1,"$(tpb.meta.name) running... ")

        minCalls=Inf
        best_hyperparam = Vector{Float64}(undef,1)
        guess_low_bound = []
        guess_up_bound = []
        dfx = DataFrame(A = x)
        f_x = Vector{Float64}(undef,grid)
        for i=1:grid

            (success,obj,cstr)=tpb.output([X[i],Y[j]])
            f_x[i]=obj

            if f_x[i]<minCalls
                minCalls=f_x[i]
                best_hyperparam=[x[i]]
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

        return (best_hyperparam,minCalls,guess_low_bound,guess_up_bound)

    end

end
