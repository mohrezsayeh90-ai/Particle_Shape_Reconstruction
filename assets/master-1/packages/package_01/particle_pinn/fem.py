"""Independent P2/P1 Taylor-Hood finite-element Navier-Stokes reference.

No PINN samples or predictions enter this solver. Both methods use the same
specified geometry, viscosity, and velocity boundary conditions.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import bmat, csr_matrix
from scipy.sparse.linalg import spsolve
from skfem import MeshTri, Basis, ElementVector, ElementTriP2, ElementTriP1, BilinearForm, asm
from skfem.helpers import dot, ddot, grad, div, mul


@BilinearForm
def viscosity(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def divergence(u, q, w):
    return q * div(u)


@BilinearForm
def advection(u, v, w):
    return dot(mul(grad(u), w.wind), v)


@dataclass
class FEMResult:
    channel: object
    re: float
    ub: object
    pb: object
    velocity: np.ndarray
    pressure: np.ndarray
    diagnostics: dict

    def evaluate(self, xy):
        xy = np.atleast_2d(xy)
        u = self.ub.interpolator(self.velocity)(xy.T).T
        p = self.pb.interpolator(self.pressure)(xy.T)
        return np.column_stack((u, p))

    def fast_field(self):
        # Interpolate in the topological (s, eta) rectangle, never across a bend's hole.
        # Dense P2 sampling, rather than physical-space Delaunay, preserves the tube.
        from scipy.interpolate import LinearNDInterpolator
        # P2 nodal values expressed in the reference rectangle. Mid-edge nodes
        # use averages of their endpoint reference coordinates, avoiding a
        # false shortcut across the physical hole of an annular channel.
        mesh = self.ub.mesh
        vs, ve = self.channel.coordinates(mesh.p.T)
        vertex_q = np.column_stack((vs, ve))
        edge_q = vertex_q[mesh.facets].mean(axis=0)
        q2 = np.vstack((vertex_q, edge_q))
        components = self.ub.split_indices()
        vel = np.column_stack((self.velocity[components[0]], self.velocity[components[1]]))
        ui = LinearNDInterpolator(q2, vel)
        pi = LinearNDInterpolator(vertex_q, self.pressure)

        def field(points):
            a, b = self.channel.coordinates(points)
            coords = np.column_stack((np.clip(a, 1e-10, self.channel.length-1e-10), np.clip(b, -.999999, .999999)))
            return np.column_stack((ui(coords), pi(coords)))
        return field


def solve_flow(channel, re=1., ns=70, nn=12, max_iter=70, tolerance=1e-8):
    xy, _, _ = channel.grid(ns, nn)
    triangles = []
    for i in range(ns-1):
        for j in range(nn-1):
            a, b = i*nn+j, (i+1)*nn+j
            triangles.extend(((a, b, b+1), (a, b+1, a+1)))
    mesh = MeshTri(xy.T, np.array(triangles).T)
    ub = Basis(mesh, ElementVector(ElementTriP2()), intorder=5)
    pb = Basis(mesh, ElementTriP1(), intorder=5)
    k, b = asm(viscosity, ub), asm(divergence, ub, pb)
    boundary = mesh.boundary_facets()
    dofs = ub.get_dofs(facets=boundary).all()
    fixed = np.r_[dofs, ub.N]  # pressure gauge only; never an artificial pressure field
    total = ub.N + pb.N
    free = np.setdiff1d(np.arange(total), fixed)
    solution = np.zeros(total)
    loc = ub.doflocs[:, dofs].T
    s, eta = channel.coordinates(loc)
    values = np.zeros((len(dofs), 2))
    cap = (s < 1e-7) | (s > channel.length-1e-7)
    _, tangents, _ = channel.frame(np.clip(s[cap], 0, channel.length))
    values[cap] = np.maximum(0, 1-eta[cap]**2)[:, None] * tangents
    # ElementVector interleaves the two velocity components.
    solution[dofs] = values[np.arange(len(dofs)), dofs % 2]
    changes = []
    nonlinear_residual = float("inf")
    for iteration in range(max_iter):
        c = asm(advection, ub, wind=ub.interpolate(solution[:ub.N]))
        a = bmat([[k + re*c, -b.T], [b, csr_matrix((pb.N, pb.N))]], format="csr")
        rhs = -a[free][:, fixed] @ solution[fixed]
        candidate = solution.copy()
        candidate[free] = spsolve(a[free][:, free], rhs)
        change = np.linalg.norm(candidate[:ub.N]-solution[:ub.N]) / max(np.linalg.norm(candidate[:ub.N]), 1e-12)
        solution[free] = .8*candidate[free] + .2*solution[free]
        changes.append(float(change))
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Non-finite FEM iterate.")
        if change < tolerance:
            break
    c = asm(advection, ub, wind=ub.interpolate(solution[:ub.N]))
    a = bmat([[k + re*c, -b.T], [b, csr_matrix((pb.N, pb.N))]], format="csr")
    nonlinear_residual = float(np.linalg.norm((a@solution)[free]) / max(np.linalg.norm((a@solution)[fixed]), 1.))
    uh = ub.interpolate(solution[:ub.N])
    divergence_l2 = float(np.sqrt(np.sum(div(uh)**2 * ub.dx)))
    diagnostics = dict(ns=ns, nn=nn, triangles=mesh.nelements, velocity_dofs=ub.N,
                       pressure_dofs=pb.N, iterations=iteration+1, relative_update=changes[-1],
                       nonlinear_residual=nonlinear_residual, divergence_l2=divergence_l2,
                       solver_ok=bool(changes[-1] < tolerance and nonlinear_residual < 1e-6),
                       updates=changes)
    return FEMResult(channel, re, ub, pb, solution[:ub.N], solution[ub.N:], diagnostics)


def exact_poiseuille(xy, half_width):
    xy = np.atleast_2d(xy)
    return np.column_stack((1-(xy[:, 1]/half_width)**2, np.zeros(len(xy)), -2*xy[:, 0]/half_width**2))


def relative_error(pred, reference):
    return float(np.linalg.norm(pred-reference)/max(np.linalg.norm(reference), 1e-12))
