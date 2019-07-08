function arglina(nvar::Int=10, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    m = 2*n
    return sum((x[i] - 2/m * sum(x[j] for j = 1:n) - 1)^2 for i = 1:n) + sum((-2/m * sum(x[j] for j = 1:n) - 1)^2 for i = n+1:m)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="arglina")
end

function arglinb(nvar::Int=10, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    m=2*n
    return sum((i * sum(j * x[j] for j = 1:n) - 1)^2 for i = 1:m)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="arglinb")
end

function arglinc(nvar::Int=10, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    m=2*n
    return 2 + sum(((i-1) * sum(j * x[j] for j = 2:n-1) - 1)^2 for i = 2:m-1)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="arglinc")
end

function arwhead(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum((x[i]^2 + x[n]^2)^2 - 4 * x[i] + 3 for i=1:n-1)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="arwhead")
end

function bdqrtic(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 5 || error("bdqrtic : nvar >= 5")

  function f(x)
    n = length(x)
    return  sum((3 - 4 * x[i])^2 + (x[i]^2 + 2 * x[i+1]^2 + 3 * x[i+2]^2 + 4 * x[i+3]^2 + 5 * x[n]^2)^2 for i=1:n-4)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="bdqrtic")
end

function beale(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return  (1.5 + x[1] * (1.0 - x[2]))^2 + (2.25 + x[1] * (1.0 - x[2]^2))^2 + (2.625 + x[1] * (1.0 - x[2]^3))^2
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="beale")
end

function broydn7d(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  n2 = max(1, div(nvar, 2))
  nvar = 2 * n2 #number of variables adjusted to be even

  function f(x)
    n = length(x)
    p=7/3
    return abs(1 - 2 * x[2] + (3 - x[1] / 2) * x[1])^p +
            sum(abs(1 - x[i-1] - 2 * x[i+1] + (3 - x[i] / 2) * x[i])^p for i=2:n-1) +
            abs(1 - x[n-1] + (3 - x[n] / 2) * x[n])^p +
            sum(abs(x[i] + x[i + n2])^p for i=1:n2)
  end

  return MPModel(nvar, f, (-1)*ones(nvar), precisions, name="broydn7d")
end

function brybnd(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  n2 = max(1, div(nvar, 2))
  nvar = 2 * n2 #number of variables adjusted to be even

  function f(x)
    n = length(x)
    p=7/3
    return sum(
                  (
                    x[i] * (2 + 5 * x[i]^2) + 1 -
                    sum(
                      x[j] * (1 + x[j])
                      for j = max(1, i-ml) : min(n, i+mu) if j != i
                    )
                  )^2 for i=1:n
            )
  end

  return MPModel(nvar, f, (-1)*ones(nvar), precisions, name="brybnd")
end

function chainwoo(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar = 4 * max(1, div(nvar, 4)) #number of variables adjusted to be a multiple of 4

  function f(x)
    n = length(x)
    return 1.0 + sum(100 * (x[2*i]   - x[2*i-1]^2)^2 + (1 - x[2*i-1])^2 +
               90 * (x[2*i+2] - x[2*i+1]^2)^2 + (1 - x[2*i+1])^2 +
               10 * (x[2*i] + x[2*i+2] - 2)^2 + 0.1 * (x[2*i] - x[2*i+2])^2 for i=1:div(n,2)-1)
  end

  return MPModel(nvar, f, (-2)*ones(nvar), precisions, name="chainwoo")
end

function chnrosnb_mod(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || ("chnrosnb_mod : nvar >= 2")

  function f(x)
    n = length(x)
    return 16 * sum((x[i-1] - x[i]^2)^2*(1.5+sin(i))^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end

  return MPModel(nvar, f, (-1)*ones(nvar), precisions, name="chnrosnb_mod")
end

function cosine(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum(cos(x[i]^2 - x[i+1] / 2) for i = 1:n-1)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="cosine")
end

function cragglvy(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("cragglvy : nvar>=2")

  function f(x)
    n = length(x)
    return sum((exp(x[2*i-1]) - x[2*i])^4 + 100 * (x[2*i] - x[2*i+1])^6 +
        (tan(x[2*i+1] - x[2*i+2]) + x[2*i+1] - x[2*i+2])^4 +
        x[2*i-1]^8 + (x[2*i+2] - 1)^2 for i = 1:div(n,2)-1)
  end

  return MPModel(nvar, f, 2*ones(nvar), precisions, name="cragglvy")
end

function dixmaane(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  m = max(1, div(nvar, 3))
  nvar = 3 * m #number of variables adjusted to be a multiple of 3

  function f(x)
    n = length(x)
    α :: Float64=1.0
    β :: Float64=0.0
    γ :: Float64=0.125
    δ :: Float64=0.125
    return 1 +
          sum(i / n * α * x[i]^2 for                 i=1:n) +
          sum(β * x[i]^2 * (x[i+1] + x[i+1]^2)^2 for i=1:n-1) +
          sum(γ * x[i]^2 * x[i+m]^4 for              i=1:2*m) +
          sum(i / n * δ * x[i] * x[i+2*m] for        i=1:m)
  end

  return MPModel(nvar, f, 2 * ones(nvar), precisions, name="dixmaane")
end

function dixmaani(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  m = max(1, div(nvar, 3))
  nvar = 3 * m #number of variables adjusted to be a multiple of 3

  function f(x)
    n = length(x)
    α :: Float64=1.0
    β :: Float64=0.0
    γ :: Float64=0.125
    δ :: Float64=0.125
    return 1 +
            sum((i / n)^2 * α * x[i]^2 for             i=1:n)   +
            sum(β * x[i]^2 * (x[i+1] + x[i+1]^2)^2 for i=1:n-1) +
            sum(γ * x[i]^2 * x[i+m]^4 for              i=1:2*m) +
            sum((i / n)^2 * δ * x[i] * x[i+2*m] for    i=1:m)
  end

  return MPModel(nvar, f, 2 * ones(nvar), precisions, name="dixmaani")
end
function dixmaanm(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  m = max(1, div(nvar, 3))
  nvar = 3 * m #number of variables adjusted to be a multiple of 3

  function f(x)
    n = length(x)
    α :: Float64=1.0
    β :: Float64=0.0
    γ :: Float64=0.125
    δ :: Float64=0.125
    return 1 +
            sum((i / n)^2 * α * x[i]^2 for                     i=1:n) +
            sum(i / n * β * x[i]^2 * (x[i+1] + x[i+1]^2)^2 for i=1:n-1) +
            sum(i / n * γ * x[i]^2 * x[i+m]^4 for              i=1:2*m) +
            sum((i / n)^2 * δ * x[i] * x[i+2*m] for            i=1:m)
  end

  return MPModel(nvar, f, 2 * ones(nvar), precisions, name="dixmaanm")
end

function dixon3dq(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + (x[n] - 1.0)^2 + sum((x[i] - x[i+1])^2 for i=2:n-1)
  end

  return MPModel(nvar, f, (-1) * ones(nvar), precisions, name="dixon3dq")
end

function dqdrtic(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum(x[i]^2 + 100 * (x[i+1]^2 + x[i+2]^2) for i=1:n-2)
  end

  return MPModel(nvar, f, 3 * ones(nvar), precisions, name="dqdrtic")
end

function dqrtic(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum((x[i] - i)^4 for i=1:n)
  end

  return MPModel(nvar, f, 2 * ones(nvar), precisions, name="dqrtic")
end

function edensch(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return 16 + sum((x[i] - 2)^4 + (x[i] * x[i+1] - 2 * x[i+1])^2 + (x[i+1] + 1)^2 for i=1:n-1)
  end

  return MPModel(nvar, f, zeros(nvar), precisions, name="edensch")
end

function eg2(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    sum(sin(x[1] + x[i]^2 - 1) for i=1:n-1) + sin(x[n]^2) / 2
  end

  return MPModel(nvar, f, zeros(nvar), precisions, name="eg2")
end

function engval1(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("engval : nvar >= 2")

  function f(x)
    n = length(x)
    return sum(
      (x[i]^2 + x[i+1]^2)^2 - 4 * x[i] + 3
      for i=1:n-1
    )
  end

  return MPModel(nvar, f, 2*ones(nvar), precisions, name="engval1")
end

function errinros_mod(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("errinros_mod : nvar >= 2")

  function f(x)
    n = length(x)
    return sum((x[i-1] - 16.0 * x[i]^2 * (1.5 + sin(i))^2)^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end

  return MPModel(nvar, f, -ones(nvar), precisions, name="errinros_mod")
end

function extrosnb(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + (1 - x[1])^2
  end

  return MPModel(nvar, f, -ones(nvar), precisions, name="extrosnb")
end

function fletcbv2(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("fletcbv2 : nvar >= 2")

  function f(x)
    n = length(x)
    h = 1.0 / (n + 1)
    return 0.5 * (x[1]^2 + sum((x[i] - x[i+1])^2 for i=1:n-1) + x[n]^2) -
    h^2 * sum(2 * x[i] + cos(x[i]) for i=1:n) - x[n]
  end

  return MPModel(nvar, f, [(i/(nvar+1.0)) for i=1:nvar], precisions, name="fletcbv2")
end

function fletcbv3_mod(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("fletcbv3_mod : nvar >= 2")

  function f(x)
    n = length(x)
    p = 10.0^(-8)
    h = 1.0 / (n + 1)
    return (p / 2.0) * (x[1]^2 + sum((x[i] - x[i+1])^2 for i=1:n-1) + x[n]^2) -
     p * sum(100.0 * (1 + (2.0 / h^2)) * sin(x[i] / 100.0) + (1 / h^2) * cos(x[i]) for i=1:n)
   end

  return MPModel(nvar, f, [(i/(nvar+1.0)) for i=1:nvar], precisions, name="fletcbv3_mod")
end

function fletchcr(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return 100 * sum((x[i+1] - x[i] + 1 - x[i]^2)^2 for i=1:n-1)
  end

  return MPModel(nvar, f, zeros(nvar), precisions, name="fletchcr")
end

function freuroth(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum(((5 - x[i+1]) * x[i+1]^2 + x[i] - 2 * x[i+1] - 13)^2 for i=1:n-1) + sum(((1 + x[i+1]) * x[i+1]^2 + x[i] - 14 * x[i+1] - 29)^2 for i=1:n-1)
  end

  x0 = zeros(nvar)
  x0[1] = 0.5
  x0[2] = -2.0
  return MPModel(nvar, f, x0, precisions, name="freuroth")
end

function genhumps(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    ζ = 20
    return sum((sin(ζ * x[i])^2 * sin(ζ * x[i+1])^2 + (x[i]^2 + x[i+1]^2) / ζ) for i=1:n-1)
  end

  x0 = -506.2 * ones(nvar)
  x0[1] = -506.0
  return MPModel(nvar, f, x0, precisions, name="genhumps")
end

function genrose(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return 1 + 100 * sum((x[i+1] - x[i]^2)^2 for i=1:n-1) + sum((x[i] - 1)^2 for  i=1:n-1)
  end

  x0 = [i / (nvar+1) for i = 1 : nvar]
  return MPModel(nvar, f, x0, precisions, name="genrose")
end

function genrose_nash(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("genrose_nash : nvar >= 2")

  function f(x)
    n = length(x)
    return 1.0 + 100 * sum((x[i] - x[i-1]^2)^2 for i=2:n) + sum((1.0 - x[i])^2 for i=2:n)
  end

  return MPModel(nvar, f, [(i/(nvar+1.0)) for i=1:nvar], precisions, name="genrose_nash")
end

function indef_mod(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 3 || error("indef_mod : nvar >= 3")

  function f(x)
    n = length(x)
    return 100.0 * sum(sin(x[i] / 100.0) for i=1:n) + 0.5 * sum(cos(2.0 * x[i] - x[n] - x[1]) for i=2:n-1)
  end

  return MPModel(nvar, f, [(i/(nvar+1.0)) for i=1:nvar], precisions, name="indef_mod")
end

function liarwhd(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("liarwhd : nvar >= 2")

  function f(x)
    n = length(x)
    return sum(4.0*(x[i]^2 - x[1])^2 + (x[i] - 1)^2  for i=1:n)
  end

  return MPModel(nvar, f, 4*ones(nvar), precisions, name="liarwhd")
end

function morebv(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("morebv : nvar >= 2")

  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((2.0 * x[i] - x[i-1] - x[i+1] + (h^2 / 2.0) * (x[i] + (i - 1) * h + 1)^3)^2 for i=2:n-1)
  end

  x0 = 0.5 * ones(nvar)
  x0[1] = 0.0
  x0[nvar] = 0.0

  return MPModel(nvar, f, x0, precisions, name="morebv")
end

function ncb20(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 20 || error("ncb20 : nvar >= 20")

  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return 2.0 +
	    sum((10.0 / i) * (sum(x[i + j - 1] / (1 + x[i + j - 1]^2) for j=1:20))^2 - 0.2 * sum(x[ i + j - 1] for j=1:20) for i=1:n-30) +
      sum(x[i]^4 + 2 for i=1:n-10) +
      1.0e-4 * sum(x[i] * x[i + 10] * x[i + n - 10] + 2.0 * x[i + n - 10]^2 for i=1:10)
  end

  x0 = ones(nvar)
  x0[1:nvar-10] .= 0.0

  return MPModel(nvar, f, x0, precisions, name="ncb20")
end

function ncb20b(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 20 || error("ncb20 : nvar >= 20")

  function f(x)
    n = length(x)
    h = 1.0/(n-1)
    return sum((10.0 / i) * (sum(x[i+j-1] / (1 + x[i+j-1]^2) for j=1:20))^2 - 0.2 * sum(x[i+j-1] for j=1:20) for i=1:n-19) +
      sum(100.0 * x[i]^4 + 2.0 for i=1:n)
  end

  return MPModel(nvar, f, zeros(nvar), precisions, name="ncb20")
end

function noncvxu2(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("noncvxu2 : nvar >= 2")

  function f(x)
    n = length(x)
    return sum((x[i] + x[mod(3 * i - 2, n) + 1] + x[mod(7 * i - 3, n) + 1])^2 +
      4.0 * cos(x[i] + x[mod(3 * i - 2, n) + 1] + x[mod(7 * i - 3, n) + 1]) for i=1:n)
  end

  return MPModel(nvar, f, Float64.([i for i=1:nvar]), precisions, name="noncvxu2")
end

function noncvxun(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("noncvxun : nvar >= 2")

  function f(x)
    n = length(x)
    return sum((x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1])^2 +
      4.0 * cos(x[i] + x[mod(2*i-1, n) + 1] + x[mod(3*i-1, n) + 1]) for i=1:n)
  end

  return MPModel(nvar, f, Float64.([i for i=1:nvar]), precisions, name="noncvxun")
end

function nondia(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("nondia : nvar >= 2")

  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((100.0*x[1] - x[i-1]^2)^2 for i=2:n)
  end

  return MPModel(nvar, f, -ones(nvar), precisions, name="nondia")
end

function nondquar(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return (x[1] - x[2])^2 + (x[n-1] - x[n])^2 + sum((x[i] + x[i+1] + x[n])^4 for i=1:n-2)
  end

  x0 = ones(nvar)
  x0[2 * collect(1:div(nvar, 2))] .= -1.0
  return MPModel(nvar, f, x0, precisions, name="nondquar")
end

function NZF1(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    nbis = max(1,div(n,13))
    n = 13*nbis
    l = div(n,13)
    return sum(
            (3*x[i] - 60 + 1/10*(x[i+1] - x[i+2])^2)^2
            + (x[i+1]^2 + x[i+2]^2 + (x[i+3]^2)*(1+x[i+3])^2 + x[i+6] + x[i+5]/(1 + x[i+4]^2 + sin(x[i+4]/1000)))^2
            + (x[i+6] + x[i+7] - x[i+8]^2 + x[i+10])^2
            + (log(1 + x[i+10]^2) + x[i+11] - 5*x[i+12] + 20)^2
            + (x[i+4] + x[i+5] + x[i+5]*x[i+9] + 10*x[i+9] - 50)^2
            for i = 1:l)
            + sum((x[i+6]-x[i+19])^2 for i=1:l-1)
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="NZF1")
end

function penalty2(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 3 || error("penalty2 : nvar >= 3")

  function f(x)
    n = length(x)
    a = 1.0e-5
    m = 2 * n
    y = ones(m)
    for i = 1:m
      y[i] = exp(i / 10.0) + exp((i-1) / 10.0)
    end
    return (x[1] - 0.2)^2 +
      sum(a * (exp(x[i] / 10.0) + exp(x[i-1] / 10.0) - y[i])^2 for i=2:n) +
      sum(a * (exp(x[i-n+1] / 10.0) - exp(-1/10))^2 for i=n+1:2*n-1) +
      (sum((n-j+1) * x[j]^2  for j=1:n) - 1.0)^2
  end

  return MPModel(nvar, f, (1/2) * ones(nvar), precisions, name="penalty2")
end

function penalty3(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 3 || error("penalty3 : nvar >= 3")

  function f(x)
    n = length(x)
    return 1.0 + sum((x[i] - 1.0)^2 for i=1:div(n,2)) +
      exp(x[n]) * sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) +
      sum((x[i] + 2.0 * x[i+1] + 10.0 * x[i+2] - 1.0)^2 for i=1:n-2) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      exp(x[n-1]) * sum((2.0 * x[i] + x[i+1] - 3.0)^2 for i=1:n-2) +
      (sum(x[i]^2 - n for i=1:n))^2
  end

  return MPModel(nvar, f, [i/(nvar+1) for i=1:nvar], precisions, name="penalty3")
end

function powellsg(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar = 4 * max(1, div(nvar, 4)) #number of variables adjusted to be a multiple of 4

  function f(x)
    n = length(x)
    return sum(-(1 / (1 + (x[i] - x[i+1])^2)) - sin((π * x[i+1] + x[i+2]) / 2) - exp(-((x[i] + x[i+2]) / x[i+1] - 2)^2) for i=1:n-2)
  end

  return MPModel(nvar, f, 3 * ones(nvar), precisions, name="powellsg")
end

function power(nvar::Int=1000, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return (sum((i * x[i]^2) for i=1:n))^2
  end

  return MPModel(nvar, f, ones(nvar), precisions, name="power")
end

function quartc(nvar::Int=1000, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum((x[i] - i)^4 for i=1:n)
  end

  return MPModel(nvar, f, 2*ones(nvar), precisions, name="quartc")
end

function sbrybnd(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("sbrybnd : nvar >= 2")
  p = zeros(nvar)
  J = Array{Any}(undef, nvar)
  for i=1:nvar
    p[i] = exp(6.0*(i-1)/(nvar-1))
    J[i] = [max(1, i-5):i-1; i+1:min(nvar, i+1)]
  end

  function f(x)
    n = length(x)
    return sum(((2.0 + 5.0 * p[i]^2 * x[i]^2) * p[i] * x[i] + 1.0 - sum(p[j] * x[j] * (1.0 + p[j] * x[j]) for j=J[i]))^2 for i=1:n)
  end

  return MPModel(nvar, f, [1/p[i] for i=1:nvar], precisions, name="sbrybnd")
end

function schmvett(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum(-(1 / (1 + (x[i] - x[i+1])^2)) - sin((π * x[i+1] + x[i+2]) / 2) - exp(-((x[i] + x[i+2]) / x[i+1] - 2)^2) for i=1:n-2)
  end

  return MPModel(nvar, f, 3 * ones(nvar), precisions, name="schmvett")
end

function scosine(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("scosine : nvar >= 2")
  p = zeros(nvar)
  for i=1:nvar
    p[i] = exp(6.0 * (i-1) / (nvar-1))
  end

  function f(x)
    n = length(x)
    return sum(cos(p[i]^2 * x[i]^2 - p[i+1] * x[i+1] / 2.0) for i=1:n-1)
  end

  return MPModel(nvar, f,[1/p[i] for i=1:nvar], precisions, name="scosine")
end

function sparsine(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 10 || error("sparsine : nvar >= 10")

  function f(x)
    n = length(x)
    return 0.5 * sum(
        i * (sin(x[i]) +
        sin(x[mod(2*i-1, n) + 1]) +
        sin(x[mod(3*i-1, n) + 1]) +
        sin(x[mod(5*i-1, n) + 1]) +
        sin(x[mod(7*i-1, n) + 1]) +
        sin(x[mod(11*i-1, n) + 1]))^2 for i=1:n)
  end

  return MPModel(nvar, f,0.5*ones(nvar), precisions, name="sparsine")
end

function sparsqur(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 10 || error("sparsqur : nvar >= 10")

  function f(x)
    n = length(x)
    return 1/8 * sum(
        i * (x[i]^2 +
        x[mod(2*i-1, n) + 1]^2 +
        x[mod(3*i-1, n) + 1]^2 +
        x[mod(5*i-1, n) + 1]^2 +
        x[mod(7*i-1, n) + 1]^2 +
        x[mod(11*i-1, n) + 1]^2)^2 for i=1:n)
  end

  return MPModel(nvar, f,0.5*ones(nvar), precisions, name="sparsqur")
end

function srosenbr(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar = 2 * max(1, div(nvar, 2)) #number of variables adjusted to be even

  function f(x)
    n = length(x)
    return sum(100.0 * (x[2*i] - x[2*i-1]^2)^2  + (x[2*i-1] - 1.0)^2 for i=1:div(n, 2))
  end

  x0 = ones(nvar)
  x0[2*(collect(1:div(nvar,2))).-1] .= -1.2

  return MPModel(nvar, f,0.5*ones(nvar), precisions, name="srosenbr")
end

function sinquad(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x) # sinquad objective
    n = length(x)
    return (x[1] - 4)^4 + (x[n]^2 - x[1]^2)^2 + sum((sin(x[i] - x[n]) - x[1]^2 + x[i]^2)^2 for i=2:n-1)
  end

  return MPModel(nvar, f, 0.1 * ones(nvar), precisions, name="sinquad")
end

function tointgss(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 3 || error("tointgss : nvar >= 3")

  function f(x)
    n = length(x)
    return sum((10.0 / (n + 2) + x[i+2]^2) * (2.0 - exp(-(x[i] - x[i+1])^2 / (0.1 + x[i+2]^2))) for i=1:n-2)
  end

  return MPModel(nvar, f,3*ones(nvar), precisions, name="tointgss")
end

function tquartic(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar >= 2 || error("tquartic : nvar >= 2")

  function f(x)
    n = length(x)
    return (x[1] - 1.0)^2 + sum((x[1]^2 - x[i+1]^2)^2 for i=1:n-2)
  end

  return MPModel(nvar, f,ones(nvar), precisions, name="tquartic")
end

function tridia(nvar::Int=10000, precisions::Vector{DataType}=builtin_fps)

  function f(x)
    n = length(x)
    α::Float64=2.0
    β::Float64=1.0
    γ::Float64=1.0
    δ::Float64=1.0
    return γ * (x[1] * δ - 1.0)^2 + sum(i * (-β * x[i-1] + α * x[i])^2 for i=2:n)
  end

  return MPModel(nvar, f,ones(nvar), precisions, name="tridia")
end

function vardim(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)
  function f(x)
    n = length(x)
    return sum((x[i] - 1)^2 for i=1:n) + sum(i * (x[i] - 1) for i=1:n)^2 + sum(i * (x[i] - 1) for i=1:n)^4
  end

  x0 = [1 - i/nvar for i = 1 : nvar]
  return MPModel(nvar, f, x0, precisions, name="vardim")
end

function woods(nvar::Int=100, precisions::Vector{DataType}=builtin_fps)

  nvar = 4 * max(1, div(nvar, 4)) #number of variables adjusted to be a multiple of 4

  function f(x)
    n = length(x)
    return 1.0 + sum(
        100 * (x[4*i-2] - x[4*i-3]^2)^2 + (1 - x[4*i-3])^2 +
        90 * (x[4*i] - x[4*i-1]^2)^2 + (1 - x[4*i-1])^2 +
        10 * (x[4*i-2] + x[4*i] - 2)^2 + 0.1 * (x[4*i-2] - x[4*i])^2 for i=1:div(n,4))
  end

  x0 = -3 * ones(nvar)
  x0[2*(collect(1:div(nvar,2)))] .= -1.0

  return MPModel(nvar, f,x0, precisions, name="woods")
end

problem_list = [arglina, arglinb, arglinc, arwhead, bdqrtic, beale, broydn7d, brybnd, chainwoo, chnrosnb_mod, cosine, cragglvy, dixmaane, dixmaani, dixmaanm, dixon3dq, dqdrtic, dqrtic, edensch, eg2, engval1, errinros_mod, extrosnb, fletcbv2, fletcbv3_mod, fletchcr,
                      freuroth, genhumps, genrose, genrose_nash, indef_mod, liarwhd, morebv, ncb20, ncb20b, noncvxu2, noncvxun, nondia, nondquar, NZF1, penalty2, penalty3, powellsg, power, quartc, sbrybnd, schmvett, scosine, sparsine, sparsqur, srosenbr, sinquad, tointgss, tquartic, tridia, vardim, woods]
