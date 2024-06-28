using HierarchicalSchrodinger, RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials
using MatrixFactorizations
import MatrixFactorizations: ldiv!
using Plots

"""
Section 6.3 "Time-dependent Schrödinger equation"

Domain is Ω = {0 ≤ r ≤ 50} and we are solving
    i ∂ₜ u(x,y,t) = (-Δ + r²) u(x,y,t), u⁽⁰⁾(x, y) = ψ_{20,21}(x, y)
on the disk with zero Dirichlet bcs at r=50, 

where ψ_{20,21} is an eigenfunction of (-Δ + r²).

The exact solution is u(x,y,t) =  exp(−i E_{20,21} t) ψ_{20,21}(x, y) where
E_{20,21} is the eigenvalue corresponding to ψ_{20,21}(x, y).

"""

# Eigenvalue parameters
nx, ny = 2, 2
E = 2*(nx+ny+1)
# Eigenfunctions of (-Δ + r²)
function ψa(xy, t)
    x, y = first(xy), last(xy)
    H = Normalized(Hermite())
    exp(-E*im*t) * H[x,nx+1] * H[y,ny+1] * exp(-(x^2+y^2)/2)
end

# Initial state at t=0
function u0(xy)
    ψa(xy, 0)
end

# Endpoints of cells in the mesh
points = [0; [50*1.2^(-n) for n in 15:-1:0]]; K = length(points)-1;
# Construct H¹ conforming disk FEM basis, truncation degree N=100
N=50; 
@time Φ = ContinuousZernike(N, points);

V(r²) = r² # quadratic well

# Analysis, compute coefficient vector of initial state
@time u0c_F = Φ \ u0.(axes(Φ,1));
# Synthesis, evaluate discretized initial state and check the error
(θs, rs, vals) = finite_plotvalues(Φ, u0c_F, N=100)
vals_, er = inf_error(Φ, θs, rs, vals, u0) # Check inf-norm errors on the grid
er

# Solve Helmholtz equation in weak form
tic = @elapsed M = Φ' * Φ; # <v, u>, v, u ∈ Φ
tic = @elapsed wM = Φ' * (V.(axes(Φ,1)) .* Φ); # <v, V(r²) u>, v, u ∈ Φ
D = Derivative(axes(Φ,1));
tic = @elapsed nΔ = (D*Φ)' * (D*Φ); # <∇v, ∇u>, v, u ∈ Φ

# Setup of time loop via a Crank-Nicolson time discretization
us = [[],[],[],[],[],[]]
# Number of timesteps from t=0 to t=2π/E
kTs = [1, 5, 10, 60, 100, 400]
for (kT, i) in zip(kTs, 1:lastindex(kTs))
    δt = 2π / E / kT # Step size
    # Crank-Nicolson matrices
    Z = sparse.(-(im * δt) .* (nΔ .+ wM))
    Ms = sparse.(M)
    Ap = 2 .* Ms .- Z
    An = 2 .* Ms .+ Z
    # Ap = sparse.(2 .*M .+ (im * δt) .* (nΔ .+ wM));
    # An = sparse.(2 .*M .- (im * δt) .* (nΔ .+ wM));

    zero_dirichlet_bcs!(Φ, Ap); # bcs
    us[i] = Any[u0c_F];
    u = u0c_F;

    lu_Ap = MatrixFactorizations.lu.(Ap);
    # @time ul_Ap = MatrixFactorizations.ul.(Ap);

    # for loop for time-stepping
    for its in 1:kT
        # explicit time half-step
        fu = (An .* u)
        zero_dirichlet_bcs!(Φ, fu);
        # implicit time half-step
        u = ldiv!.(lu_Ap, fu)
        append!(us[i], [u])
        print("Time step: $its \n")
    end
end

errs = []
for i = 1:6
    its = lastindex(us[i])
    # Sythesis (expansion) for real and complex-valued coefficients
    (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us[i][its]), N=100);
    (_, _, vals_im) = finite_plotvalues(Φ, imag.(us[i][its]), N=100);
    t = 2π/E

    # tdisplay = round(t, digits=4)
    # SparseDiskFEM.plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
    # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

    # Compute the ℓ^∞ error
    ua(xy) = ψa(xy,t)
    vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua)
    append!(errs, err)
    print("error: $(errs[end]) \n")
    # writedlm("errors-schrodinger-pade.log", errs)
end

us_p = [[],[],[],[],[],[]]
# Number of timesteps from t=0 to t=2π/E
kTs = [1, 5, 10, 60, 100, 400]
for (kT, i) in zip(kTs, 1:lastindex(kTs))
    δt = 2π / E / kT # Step size
    # Crank-Nicolson matrices
    Z = - (im * δt) .*  sparse.((nΔ .+ wM))
    a, b = 3.0, sqrt(3.0)
    Ap1 = sparse.(Z .- (a - im*b).*M);
    Ap2 = sparse.(Z .- (a + im*b).*M);
    An1 = sparse.(Z .+ (a - im*b).*M);
    An2 = sparse.(Z .+ (a + im*b).*M);

    zero_dirichlet_bcs!(Φ, Ap1); # bcs
    zero_dirichlet_bcs!(Φ, Ap2); # bcs

    us_p[i] = Any[u0c_F];
    u = u0c_F;

    lu_Ap1 = MatrixFactorizations.lu.(Ap1)
    lu_Ap2 = MatrixFactorizations.lu.(Ap2)

    # for loop for time-stepping
    for its in 1:kT
    # its =  1
        # explicit time half-step
        fu = copy(An1 .* u)
        zero_dirichlet_bcs!(Φ, fu);

        u = copy(ldiv!.(lu_Ap1, fu))

        fu = copy(An2 .* u)
        zero_dirichlet_bcs!(Φ, fu);
        # implicit time half-step
        
        u = copy(ldiv!.(lu_Ap2, fu))

        append!(us_p[i], [u])
        print("Time step: $its \n")
    end
end


errs_p = []
for i = 1:6
    its = lastindex(us_p[i])
    # Sythesis (expansion) for real and complex-valued coefficients
    (θs, rs, vals_r) = finite_plotvalues(Φ, real.(us_p[i][its]), N=100);
    (_, _, vals_im) = finite_plotvalues(Φ, imag.(us_p[i][its]), N=100);
    t = 2π/E

    # tdisplay = round(t, digits=4)
    # SparseDiskFEM.plot(Φ, θs, rs, vals_r, ttl=L"$\mathrm{Re} \; u(x,y,%$(tdisplay))$", vminmax=[-0.4,0.4], K=Kp) # plot
    # PyPlot.savefig("examples/plots-harmonic-oscillator/$i.png")

    # Compute the ℓ^∞ error
    ua(xy) = ψa(xy,t)
    vals_, err = inf_error(Φ, θs, rs, vals_r .+ im.*vals_im, ua)
    append!(errs_p, err)
    print("error: $(errs_p[end]) \n")
    writedlm("errors-schrodinger-pade.log", errs)
end


l2_norms = []
for its in 1:401
    append!(l2_norms, [sqrt(abs(sum(adjoint.(us_p[6][its]) .* (M .* us_p[6][its]))))])
    print("Time step: $its \n")
end

# Difference in L^2-norm with initial state
l2_diff = abs.(l2_norms .- l2_norms[1])
Plots.plot((2:401), l2_diff[2:401],
    linewidth=2,
    ylabel=L"$|\Vert u^{(k)} \Vert_{L^2(\Omega)} - \Vert u^{(0)}  \Vert_{L^2(\Omega)}| $",
    xlabel=L"$k$",
    gridlinewidth = 2,
    tickfontsize=10, ytickfontsize=10,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize=10, titlefontsize=20,
    markersize = 2,
    marker=:circle,
    legend=:none,
    margin=4Plots.mm
)

###
# Convergence plots
###
δts = 2π / E ./ kTs
# Expected second-order convergence
second_order_rate = δts.^2 ./ δts[2]^2 / 3 * errs[1]
fourth_order_rate = δts.^4 ./ δts[2]^2 / 3 * errs_p[1]
Plots.plot(δts, Float64.(hcat(errs, errs_p, second_order_rate, fourth_order_rate)),
    label=["Crank-Nicolson" "(2,2) Pade" L"O(\delta t^2)" L"O(\delta t^4)"],
    linewidth=[3 3 2 2],
    markershape=[:circle :diamond :none :none],
    linestyle=[:solid :solid :dash :dash],
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\delta t$",
    legend=:bottomright,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize = 13,
    yscale=:log10,
    xscale=:log10,
    # yticks=[1e-6, 1e-4, 1e-2, 1e0],
    # xticks=[1e-4,1e-3,1e-2,1e-1],
    # xlim=[5e-5,1e-1],
    # ylim=[1e-6,1e0],
    gridlinewidth = 2,
    margin=4Plots.mm,
)
Plots.savefig("schrodinger-convergence.pdf")


Plots.plot(δts, Float64.(hcat(errs_p, fourth_order_rate)),
    label=["" L"O(\delta t^4)"],
    linewidth=[3 2],
    markershape=[:circle :none],
    linestyle=[:solid :dash],
    markersize=5,

    ylabel=L"$l^\infty\mathrm{-norm \;\; error}$",
    xlabel=L"$\delta t$",
    legend=:bottomright,
    xtickfontsize=12, ytickfontsize=12,xlabelfontsize=15,ylabelfontsize=15,
    legendfontsize = 13,
    yscale=:log10,
    xscale=:log10,
    # yticks=[1e-6, 1e-4, 1e-2, 1e0],
    # xticks=[1e-4,1e-3,1e-2,1e-1],
    # xlim=[5e-5,1e-1],
    # ylim=[1e-6,1e0],
    gridlinewidth = 2,
    margin=4Plots.mm,
)
Plots.savefig("schrodinger-convergence.pdf")