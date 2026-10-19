"""Automatic steering: Kalman-filtered lane state + lateral MPC.

State (relative to the lane):  y   lateral position, + right of center [m]
                               psi heading vs. lane, + left [rad]
Input:                         delta road-wheel angle, + left [rad]
Linearized kinematic bicycle:  y'   = -v * psi
                               psi' =  v / L * delta - v * kappa

MPC: every frame, choose delta_0..delta_{N-1} over a 3 s horizon minimizing
    sum  q_y*y^2 + q_psi*psi^2              (stay centered and aligned)
       + r*(delta - delta_ff)^2             (no steering beyond what the curve needs)
       + r_rate*(delta_k - delta_{k-1})^2   (smooth wheel)
subject to |delta| <= max_delta and |d delta/dt| <= max_rate,
then apply delta_0 only (receding horizon). The QP is solved with OSQP.
"""
import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

# numpy 2.0 + macOS Accelerate raises spurious "divide by zero / invalid value in
# matmul" warnings even for finite inputs; results are checked with isfinite instead.
quiet_fp = np.errstate(divide="ignore", over="ignore", invalid="ignore")


class LaneKF:
    """Constant-model Kalman filter over [y, psi] driven by the applied steering."""

    def __init__(self, std_y=0.25, std_psi=np.radians(1.5), proc_y=0.02, proc_psi=np.radians(0.3)):
        self.R = np.diag([std_y ** 2, std_psi ** 2])
        self.Q = np.diag([proc_y ** 2, proc_psi ** 2])
        self.reset()

    def reset(self):
        self.x = None
        self.P = self.R.copy()

    def predict(self, v, delta, kappa, dt, wheelbase):
        if self.x is None:
            return
        y, psi = self.x
        self.x = np.array([y - v * np.sin(psi) * dt, psi + (v / wheelbase * np.tan(delta) - v * kappa) * dt])
        A = np.array([[1.0, -v * dt], [0.0, 1.0]])
        self.P = A @ self.P @ A.T + self.Q

    def update(self, z):
        if self.x is None:
            self.x, self.P = z.copy(), self.R.copy()
            return
        S = self.P + self.R
        innov = z - self.x
        if innov @ np.linalg.solve(S, innov) > 30:  # lane slot switched (line crossing): re-acquire
            self.x, self.P = z.copy(), self.R.copy()
            return
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ innov
        self.P = (np.eye(2) - K) @ self.P


@dataclass
class Plan:
    delta: float          # command to apply now [rad]
    deltas: np.ndarray    # planned road-wheel angles over the horizon [rad]
    path: np.ndarray      # predicted (X, Y) in the current vehicle frame, (N+1, 2) [m]
    times: np.ndarray     # time of each path point [s]
    y_pred: np.ndarray    # predicted lateral offset vs. lane [m]
    solve_ms: float
    status: str


class LateralMPC:
    def __init__(self, wheelbase, horizon=30, dt=0.1, ctrl_dt=1 / 30,
                 q_y=3.0, q_psi=8.0, q_term=5.0, r=40.0, r_rate=400.0,
                 max_delta=np.radians(8), max_rate=np.radians(10)):
        self.L, self.N, self.dt, self.ctrl_dt = wheelbase, horizon, dt, ctrl_dt
        self.q = np.array([q_y, q_psi])
        self.q_term, self.r, self.r_rate = q_term, r, r_rate
        self.max_delta, self.max_rate = max_delta, max_rate

        N = horizon
        self.D = (sp.eye(N) - sp.eye(N, k=-1)).toarray()   # rate rows: u_k - u_{k-1}
        A = sp.vstack([sp.eye(N), sp.csc_matrix(self.D)]).tocsc()
        pattern = sp.triu(sp.csc_matrix(np.ones((N, N)))).tocsc()
        self.p_rows, self.p_cols = pattern.nonzero()
        order = np.lexsort((self.p_rows, self.p_cols))      # CSC data order: by column, then row
        self.p_rows, self.p_cols = self.p_rows[order], self.p_cols[order]
        self.solver = osqp.OSQP()
        self.solver.setup(P=pattern, q=np.zeros(N), A=A, l=-np.ones(2 * N), u=np.ones(2 * N),
                          verbose=False, warm_starting=True, polishing=False,
                          eps_abs=1e-6, eps_rel=1e-6, max_iter=4000)

    def _prediction(self, v, kappa):
        """x_k = Sx x0 + Su U + Sw  for k = 1..N  (exact ZOH discretization).
        kappa: scalar, or one road curvature per horizon step (preview)."""
        N, dt, L = self.N, self.dt, self.L
        A = np.array([[1.0, -v * dt], [0.0, 1.0]])
        B = np.array([-v * dt ** 2 / 2 * v / L, v / L * dt])
        kap = np.broadcast_to(np.asarray(kappa, float), (N,))
        Sx = np.zeros((2 * N, 2))
        Su = np.zeros((2 * N, N))
        Sw = np.zeros(2 * N)
        Ak = np.eye(2)
        w = np.zeros(2)
        for k in range(N):
            # x_{k+1} = A x_k + B u_k + E
            prev = slice(2 * (k - 1), 2 * k)
            cur = slice(2 * k, 2 * k + 2)
            Ak = A @ Ak
            Sx[cur] = Ak
            if k > 0:
                Su[cur, :k] = A @ Su[prev, :k]
                w = A @ w
            Su[cur, k] = B
            w = w + np.array([v * dt ** 2 / 2 * v * kap[k], -v * kap[k] * dt])
            Sw[cur] = w
        return Sx, Su, Sw

    @quiet_fp
    def solve(self, x0, v, kappa, u_prev):
        t0 = time.perf_counter()
        N = self.N
        v = max(v, 0.5)                                  # creeping in traffic: keep the model well-posed
        u_ff = np.arctan(self.L * np.broadcast_to(np.asarray(kappa, float), (N,)))  # steering that holds the curve
        u_prev = float(np.clip(u_prev, -self.max_delta, self.max_delta))
        Sx, Su, Sw = self._prediction(v, kappa)

        qdiag = np.tile(self.q, N)
        qdiag[-2:] *= self.q_term
        free = Sx @ x0 + Sw
        e0 = np.zeros(N)
        e0[0] = 1.0
        P = 2 * (Su.T @ (qdiag[:, None] * Su) + self.r * np.eye(N) + self.r_rate * self.D.T @ self.D)
        q = 2 * (Su.T @ (qdiag * free) - self.r * u_ff * np.ones(N) - self.r_rate * self.D.T @ (e0 * u_prev))

        step = np.full(N, self.max_rate * self.dt)
        step[0] = self.max_rate * self.ctrl_dt           # first move happens within one control tick
        l = np.r_[np.full(N, -self.max_delta), -step + e0 * u_prev]
        u = np.r_[np.full(N, self.max_delta), step + e0 * u_prev]
        self.solver.update(Px=P[self.p_rows, self.p_cols], q=q, l=l, u=u)
        res = self.solver.solve()
        status = res.info.status
        U = res.x if res.x is not None and np.all(np.isfinite(res.x)) else np.full(N, u_prev)

        y_pred = np.r_[x0[0], (free + Su @ U)[0::2]]
        path = self._rollout(U, v)
        return Plan(delta=float(U[0]), deltas=U, path=path, times=np.arange(N + 1) * self.dt,
                    y_pred=y_pred, solve_ms=(time.perf_counter() - t0) * 1000, status=status)

    def _rollout(self, U, v):
        """Planned motion in the current vehicle frame (X right, Y forward)."""
        X, Y, th = 0.0, 0.0, 0.0
        pts = [(X, Y)]
        for d in U:
            X -= v * np.sin(th) * self.dt
            Y += v * np.cos(th) * self.dt
            th += v / self.L * np.tan(d) * self.dt
            pts.append((X, Y))
        return np.array(pts)


class AutoSteer:
    """Perception (or oracle) -> filtered lane state -> MPC -> steering-wheel angle."""

    def __init__(self, wheelbase, steer_ratio, dt=1 / 30):
        self.L, self.ratio, self.dt = wheelbase, steer_ratio, dt
        self.kf = LaneKF()
        self.mpc = LateralMPC(wheelbase)
        self.kappa = 0.0
        self.delta = 0.0

    def reset(self, wheel_deg=0.0):
        self.kf.reset()
        self.delta = np.radians(wheel_deg / self.ratio)

    def __call__(self, perc, v, oracle=None):
        """perc: LaneState or None. oracle: (y, psi, kappa) to bypass perception."""
        if oracle is not None:
            x0, kappa = np.array(oracle[:2]), oracle[2]
        else:
            if perc is not None:
                self.kappa = 0.9 * self.kappa + 0.1 * float(np.clip(perc.curvature, -0.01, 0.01))
            self.kf.predict(v, self.delta, self.kappa, self.dt, self.L)
            if perc is not None:
                self.kf.update(np.array([perc.offset, -perc.heading]))
            if self.kf.x is None:
                return None
            x0, kappa = self.kf.x, self.kappa
        plan = self.mpc.solve(x0, v, kappa, self.delta)
        self.delta = plan.delta
        return plan

    def wheel_deg(self):
        return np.degrees(self.delta) * self.ratio
