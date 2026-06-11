import numpy as np
from scipy.optimize import root
from scipy.optimize import least_squares


class CorrectedLiquidTumor:

    def __init__(self, NI, NHK, delta_mu, r, r0):

        self.NI = NI
        self.NHK = NHK
        self.delta_mu = delta_mu

        self.r = np.asarray(r, dtype=float)
        self.r0 = float(r0)

        self.beta_k = self.r / (self.r + self.r0)

    # ---------------------------------------------------------
    # Mutation probabilities
    # ---------------------------------------------------------

    def sigma(self, k):
        mu = k * self.delta_mu
        return (1.0 - mu) ** self.NHK

    def delta(self, k):
        return 1.0 - self.sigma(k)

    def alpha(self, k):
        mu = k * self.delta_mu
        return (1.0 - mu) ** (self.NHK + self.NI - k)

    def gamma(self, k):

        if k >= self.NI:
            return 0.0

        mu = k * self.delta_mu

        return (
            (self.NI - k)
            * mu
            * (1.0 - mu) ** (self.NHK + self.NI - k - 1)
        )

    # ---------------------------------------------------------
    # Moran probabilities
    # ---------------------------------------------------------

    def beta_kj(self, k, j):

        rk = self.r[k - 1]
        rj = self.r[j - 1]

        if k == j:
            return 0.5

        return rk / (rk + rj)

    # ---------------------------------------------------------
    # Colonisation factors
    # ---------------------------------------------------------

    def compute_phi(self, f, pD):

        fW = 1.0 - pD - np.sum(f)
        phi = np.zeros(self.NI)
        for k in range(1, self.NI + 1):
            value = pD
            value += fW * self.beta_k[k - 1]
            for j in range(1, self.NI + 1):
                value += f[j - 1] * self.beta_kj(k, j)
            phi[k - 1] = value
        return phi, fW

    # ---------------------------------------------------------
    # Steady-state residuals
    # ---------------------------------------------------------

    def residuals(self, x):

        f = x[:-1]
        pD = x[-1]
        phi, fW = self.compute_phi(f, pD)
        res = np.zeros(self.NI + 1)
        # -----------------------------------
        # Cancer classes
        # -----------------------------------

        for k in range(1, self.NI + 1):
            fk = f[k - 1]
            phi_self = fk / 2.0
            phi_nonself = phi[k - 1] - phi_self
            ak = self.alpha(k)
            dk = self.delta(k)
            gk = self.gamma(k)
            # original cancer dynamics
            cancer_term = (
                self.r[k - 1]
                * fk
                * (
                    phi_nonself * ak
                    - phi_self * dk
                    - phi[k - 1] * gk
                )
            )

            # influx from lower class

            influx = 0.0
            if k > 1:
                influx = (
                    self.r[k - 2]
                    * f[k - 2]
                    * phi[k - 2]
                    * self.gamma(k - 1)
                )

            # WT killing class k
            wt_attack = (
                self.r0
                * fW
                * fk
                * (1.0 - self.beta_k[k - 1])
            )
            res[k - 1] = cancer_term + influx - wt_attack

        # -----------------------------------
        # Dead-cell balance
        # -----------------------------------

        dead_balance = 0.0

        for k in range(1, self.NI + 1):
            fk = f[k - 1]
            dead_balance += (
                self.r[k - 1]
                * fk
                * (
                    phi[k - 1] * self.delta(k)
                    - pD * self.sigma(k)
                )
            )

        # WT repairs dead sites

        dead_balance -= self.r0 * fW * pD
        res[-1] = dead_balance
        return res

    # ---------------------------------------------------------
    # Solve
    # ---------------------------------------------------------


    def solve(self, guess=None):

        if guess is None:
            guess = np.zeros(self.NI + 1)
            guess[0]=0.1


        sol = least_squares(self.residuals, guess, bounds=(0.0, 1.0), method="trf", xtol=1e-10, ftol=1e-10)

        return sol



# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    NI = 10
    NHK = 10

    delta_mu = 0.025

    r0 = 0.15

    r = [r0*2]*NI

    model = CorrectedLiquidTumor(
        NI=NI,
        NHK=NHK,
        delta_mu=delta_mu,
        r=r,
        r0=r0
    )

    sol = model.solve()

    if not sol.success:
        print(sol.message)
        raise RuntimeError("Root finder failed")

    f = sol.x[:-1]
    pD = sol.x[-1]

    fW = 1.0 - pD - np.sum(f)

    print("\nStationary fractions\n")

    for k, value in enumerate(f, start=1):
        print(f"f_{k} = {value:.8f}")

    print(f"\nf_W = {fW:.8f}")
    print(f"p_D = {pD:.8f}")

    print("\nNormalization:")
    print(np.sum(f) + fW + pD)