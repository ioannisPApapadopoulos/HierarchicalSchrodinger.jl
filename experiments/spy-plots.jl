using HierarchicalSchrodinger, RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials
using MatrixFactorizations
import MatrixFactorizations: ldiv!
import ClassicalOrthogonalPolynomials: ClosedInterval, colsupport
using Plots

points = [0; [10*1.2^(-n) for n in 30:-1:0]]; K = length(points)-1;
Φ = ContinuousZernike(50, points);

# V(r²) = r²

V(r²) = -1.0/sqrt(r²) # Coulomb potential
# points = [1.5^(-n) for n in 70:-1:0]
vs = Any[]
Ts = Any[]
for i in 1:lastindex(points)-1
    T = chebyshevt(points[i]..points[i+1])
    append!(Ts, [T])
    append!(vs, [T[:,1:3] \ V.(axes(T,1))])
    print("Completed interval $i \n")
end
maximum(last.(colsupport.(vs)))

intervals = ClosedInterval.(points[1:end-1],points[2:end])
function Vf(r²::Float64, intervals::AbstractVector{ClosedInterval{Float64}}, vs::AbstractArray{Any}, Ts)
    idx = findall(r² .∈ intervals)[1]
    (Ts[idx][:,1:3]*vs[idx])[r²]
end
Vf(0.3, intervals, vs, Ts)
Vff(r²) = Vf(r², intervals, vs, Ts)

tic = @elapsed M = Φ' * Φ; # <v, u>, v, u ∈ Φ
tic = @elapsed wM = Φ' * (Vff.(axes(Φ,1)) .* Φ); # <v, V(r²) u>, v, u ∈ Φ
D = Derivative(axes(Φ,1));
tic = @elapsed nΔ = (D*Φ)' * (D*Φ); # <∇v, ∇u>, v, u ∈ Φ

δt = 0.1


# Crank-Nicolson
Z = sparse.(-(im * δt) .* (nΔ .+ wM))
Ms = sparse.(M)
Ap = 2 .* Ms .- Z

zero_dirichlet_bcs!(Φ, Ap); # bcs

@time lu_Ap = MatrixFactorizations.lu.(Ap);
@time ul_Ap = MatrixFactorizations.ul.(Ap);

AA = abs.(lu_Ap[80].U)
AA[AA.>1e-15] .= 1
Plots.spy(AA)

BB = abs.(ul_Ap[80].U)
BB[BB.>1e-15] .= 1
BB[BB.<1e-15] .= 0
Plots.spy(BB)

# (2,2) Pade
Z = - (im * δt) .*  sparse.((nΔ .+ wM))
a, b = 3.0, sqrt(3.0)
Ap1 = sparse.(Z .- (a - im*b).*M);
Ap2 = sparse.(Z .- (a + im*b).*M);

zero_dirichlet_bcs!(Φ, Ap1); # bcs
zero_dirichlet_bcs!(Φ, Ap2); # bcs

@time lu_Ap = MatrixFactorizations.lu.(Ap1);
@time ul_Ap = MatrixFactorizations.ul.(Ap1);

CC = abs.(Ap1[80]);
CC[CC.>1e-15] .= 1
Plots.spy(CC)

AA = abs.(lu_Ap[80].U)
AA[AA.>1e-15] .= 1
Plots.spy(AA)

BB = abs.(ul_Ap[80].U)
BB[BB.>1e-15] .= 1
BB[BB.<1e-15] .= 0
BB = sparse(BB)
Plots.spy(BB)