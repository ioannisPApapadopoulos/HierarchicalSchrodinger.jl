using HierarchicalSchrodinger, RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials
import ClassicalOrthogonalPolynomials: ClosedInterval, colsupport
using PyPlot, Plots
"""
Solving Schroedinger's equation:
    i ∂ₜ u(x,y,t) = (-Δ + V(r)) u(x,y,t)

on the unit disk with zero Dirichlet bcs at r=1.

Here we pick:
    V(r) = 1/r
and u₀(x,y) = exp(-30((x-1/2)²+y²)) + exp(-30(x²+y²))
"""

function u0(xy)
    x,y = first(xy), last(xy)
    exp(-40*((x-0.5)^2+(y)^2))
end

points = [50*1.2^(-n) for n in 60:-1:0]; K = length(points)-1
N=80; Φ = ContinuousZernike(N, points); Ψ = ZernikeBasis(N, points, 0, 0)

V(r²) = -1.0/sqrt(r²) # Coulomb potential


# points = [1.5^(-n) for n in 70:-1:0]
vs = Any[]
Ts = Any[]
for i in 1:lastindex(points)-1
    T = chebyshevt(points[i]..points[i+1])
    append!(Ts, [T])
    append!(vs, [T[:,1:4] \ V.(axes(T,1))])
    print("Completed interval $i \n")
end
maximum(last.(colsupport.(vs)))

intervals = ClosedInterval.(points[1:end-1],points[2:end])
function Vf(r²::Float64, intervals::AbstractVector{ClosedInterval{Float64}}, vs::AbstractArray{Any}, Ts)
    idx = findall(r² .∈ intervals)[1]
    (Ts[idx][:,1:4]*vs[idx])[r²]
end
Vf(0.3, intervals, vs, Ts)
Vff(r²) = Vf(r², intervals, vs, Ts)

# v1s = Any[]
# for i in 1:lastindex(points)-1
#     T = chebyshevt(points[i]..points[i+1])
#     append!(v1s, [T \ Vff.(axes(T,1))])
#     print("Completed interval $i \n")
# end
# maximum(last.(colsupport.(v1s)))

xx = -1:0.01:1;
Plots.plot(xx, V.(xx.^2),
    linewidth=2,
    ylabel=L"$V(r)$",
    xlabel=L"$r$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    legend=:none,
    title=L"$\theta=0$",
)
Plots.savefig("examples/plots/V.png")
# x = axes(Φ,1)
# u0c = Ψ \ u0.(x)
# (θs, rs, vals) = finite_plotvalues(Ψ, u0c)
# vals_, err = inf_error(Ψ, θs, rs, vals, u0) # Check inf-norm errors on the grid
# err
# plot(Ψ, θs, rs, vals)

x = axes(Φ,1)
u0c_F = Φ \ u0.(x)
(θs, rs, vals) = finite_plotvalues(Φ, u0c_F)
vals_, err = inf_error(Φ, θs, rs, vals, u0) # Check inf-norm errors on the grid
err
plot(Φ, θs, rs, vals, K=10)


# Solve Helmholtz equation in weak form
tic = @elapsed wM = Φ' * (Vff.(x) .* Φ); # list of weighted mass matrices for each Fourier mode
# pwM = piecewise_constant_assembly_matrix(Φ, V)
tic = @elapsed M = Φ' * Φ;
D = Derivative(axes(Φ,1));
tic = @elapsed nΔ = (D*Φ)' * (D*Φ); # list of stiffness matrices for each Fourier mode
# R = Φ'*Ψ;

sM = sparse.(M);
swM = sparse.(wM);
sΔ = sparse.(nΔ);

δt = 0.001
@time Ap = 2 .*sM .+ (im * δt) .* (sΔ .+ swM);
@time An = 2 .*sM .- (im * δt) .* (sΔ .+ swM);

# UL factors are sparse, no pivoting required!
# B = [As[end:-1:1, end:-1:1] for As in A]
# LbUb = [lu(Bs, NoPivot()) for Bs in B]
# Lb = [L.L for L in LbUb]
# Ub = [U.U for U in LbUb]
# norm(B .- Lb.*Ub, Inf)
# L = [Ubs[end:-1:1, end:-1:1] for Ubs in Ub];
# U = [Lbs[end:-1:1, end:-1:1] for Lbs in Lb];
# norm(A .- U.*L, Inf)


# Mu0 = R .* u0c; # right-hand side
zero_dirichlet_bcs!(Φ, Ap); # bcs

us = AbstractVector{<:AbstractVector}[u0c_F];
u = u0c_F;

# function zero_dirichlet_bcs(Φ::ContinuousZernike{T}, Mf::AbstractVector{<:AbstractVector}) where T
#     @assert length(Mf) == 2*Φ.N-1
#     Fs = Φ.Fs #_getFs(Φ.N, Φ.points)
#     zero_dirichlet_bcs.(Fs, Mf)
# end

# function zero_dirichlet_bcs(Φ::ContinuousZernikeMode{T}, Mf::AbstractVector) where T
#     points = Φ.points
#     K = length(points)-1
#     if !(first(points) ≈  0)
#         Mf[1] = zero(T)
#         Mf[K+1] = zero(T)
#     else
#         Mf[K] = zero(T)
#     end
# end

using MatrixFactorizations
import MatrixFactorizations: ldiv!
@elapsed lu_Ap = MatrixFactorizations.lu.(Ap)

@elapsed for its in 1:3000
    fu = (An .* u)
    zero_dirichlet_bcs!(Φ, fu);
    u = ldiv!.(lu_Ap, fu)
    # u1 = Ap .\ fu
    push!(us, u)
    print("Time step: $its \n")
end

l2_norms = []
# h1_norms = []
# H = nΔ .+ M;
for its in 1:3001
    append!(l2_norms, [sqrt(abs(sum(adjoint.(us[its]) .* (sM .* us[its]))))])
    # append!(h1_norms, [sqrt(abs(sum(adjoint.(us[its]) .* (H .* us[its]))))])
    print("Time step: $its \n")
end

Plots.plot(δt.*(1:3001), l2_norms,
    linewidth=2,
    ylabel=L"$\Vert u(\cdot,\cdot,t \Vert_{L^2(\Omega)}$",
    xlabel=L"$t$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    # linestyle = :dot,
    markersize = 2,
    marker=:circle,
    legend=:none
)
Plots.savefig("examples/plots/l2-norm.png")

# using LaTeXStrings
j = 2104
for its in 2105:3001
# its = 3001
    j+=1
    tic1 = @elapsed (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[its]), K=50, N=50);
    tic2 = @elapsed (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[its]), K=50, N=50);
    vals = [(abs.(vals_r[j] + im * vals_im[j])).^2 for j in 1:lastindex(vals_r)]
    t = round(((its-1)*δt), digits=3)
    plot(Φ, θs, rs, vals, K=50, ttl=L"$|u(x,y,%$t)|^2$") # plot
    PyPlot.savefig("examples/plots/$j.png", dpi=200)
    print("Generating image: $j. Time elapsed: $(tic1+tic2) \n")
end

# ffmpeg -framerate 8 -i %d.png -pix_fmt yuv420p out.mp4

PyPlot.savefig("examples/plots/302.png", dpi=200)