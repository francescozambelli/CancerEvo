# Tumor stability is driven by stationary ecological processes

The value of the mutation rate around which the system stabilizes appears to emerge from intrinsic system dynamics, shaped by evolutionary pressures. We focus on a regime where both the replication rate and the mutation level of housekeeping genes have reached saturation, which is a common occurrence in the model, due to selective advantages associated with high proliferation and early accumulation of $\mathcal{H}$ genes mutations. 
At this stage, tumor cells can be conceptualized as belonging to distinct subpopulations, each defined by the number of instability genes $\mathcal{I}$ with a mutated phenotype. Since a cell must have at least one mutated $\mathcal{I}$ gene to be classified as cancerous, the population can be partitioned into $N_{\mathcal{I}}$ classes, labeled $x_1, \dots, x_{N_{\mathcal{I}}}$, where $x_i$ denotes the fraction of cells with $i$ mutated $\mathcal{I}$ genes. The sum over all populations is conserved, i.e.,

$$\sum_{i=1}^{N_{\mathcal{I}}} x_i = 1$$

Transitions between subpopulations are driven by three key processes: 
1) reproduction, where a cell replicates, and the daughter replaces another cell, 
2) death, due to further mutation in unmutated $\mathcal{H}$ alleles, 
3) mutation in a non-mutated $\mathcal{I}$ gene pushing the cell into a higher instability class.

The dynamics can be described using a Master Equation (ME) approach. The temporal evolution of the population densities is given by:
$$
\begin{aligned}
\frac{dx_1}{dt} &= r x_1 \big(1 - p_{\mathrm{d},1} - p_{\mu,1}\big) - \Phi x_1, \\[4pt]
\frac{dx_i}{dt} &= r x_i \big(1 - p_{\mathrm{d},i} - p_{\mu,i}\big) + r x_{i-1} p_{\mu,i-1} - \Phi x_i, \\[4pt]
\frac{dx_{N_{\mathcal{I}}}}{dt} &= r x_{N_{\mathcal{I}}} \big(1 - p_{\mathrm{d},N_{\mathcal{I}}}\big) + r x_{N_{\mathcal{I}-1}} p_{\mu,N_{\mathcal{I}-1}} - \Phi x_{N_{\mathcal{I}}}.
\end{aligned}
$$
(with $2 \le i \le N_{\mathcal{I}}-1$).

In compact vector form, this system can be written as a replication-mutation model:
$$
\frac{d\mathbf{x}}{dt} = r \mathbf{A}\mathbf{x} - \Phi \mathbf{x},
$$
where $\mathbf{x} = (x_1, x_2, \ldots, x_{N_{\mathcal{I}}})^{\mathrm{T}}$ and the transition matrix $\mathbf{A}$ has the tridiagonal structure:
$$
\mathbf{A} =
\begin{pmatrix}
1 - p_{\mathrm{d},1} - p_{\mu,1} & 0 & 0 & \cdots & 0 \\[4pt]
p_{\mu,1} & 1 - p_{\mathrm{d},2} - p_{\mu,2} & 0 & \cdots & 0 \\[4pt]
0 & p_{\mu,2} & 1 - p_{\mathrm{d},3} - p_{\mu,3} & \cdots & 0 \\[4pt]
\vdots & \vdots & \vdots & \ddots & \vdots \\[4pt]
0 & 0 & 0 & p_{\mu,N_{\mathcal{I}}-1} & 1 - p_{\mathrm{d},N_{\mathcal{I}}}
\end{pmatrix}.
$$

where we define 

$$ 
p_{\mathrm{d},i} = 1 - (1 - i \cdot \Delta\mu)^{N_{\mathcal{H}}} 
$$ 

as the death probability for cells in class $i$ and $\Phi$ is a normalization factor that enforces constant total population size over time. To calculate the probability of transition from a class $i$ to the successive one, we consider the limiting case in which each instability gene has at maximum 1 allele not mutated. This is a case in which the system eventually will end up in the long run, because the accumulation of mutation are insensible to the phenotypic outcome for the AND gate logic governing it. So, in such a limiting case, the transition probability from class $i$ to $i+1$ is given by: 

$$
p_{\text{mut},i} = 1 - (1 - i \cdot \Delta\mu)^{(N_{\mathcal{I}} - i)}
$$ 

In this model, transitions are only allowed from class $i$ to $i+1$; back mutations are not allowed, and jumps of more than one class are considered as higher order effects.


## Analytical derivation of stationary mutational populations densities

The continuous-time Master Equation governing the tumor subpopulation fractions $\mathbf{x}$ is initially non-linear:

$$\frac{d\mathbf{x}}{dt} = r\mathbf{A}\mathbf{x} - \Phi \mathbf{x}$$

This non-linearity arises from the constraint term $\Phi$, which acts as the average fitness of the entire tumor population to enforce a constant total size. Analytically, $\Phi$ is the sum of the effective growth rates across all subpopulations:

$$\Phi = r \sum_{i=1}^{N_{\mathcal{I}}} x_i (1 - p_{\mathrm{d},i})$$

Because the model dictates unidirectional transitions (from class $i$ to $i+1$) and forbids back mutations, the transition matrix $\mathbf{A}$ is strictly lower bidiagonal. For any bidiagonal matrix, the eigenvalues are simply the entries on the main diagonal. Thus, the exact eigenvalues are:
$$\lambda_i = 1 - p_{\mathrm{d},i} - p_{\mu,i}$$
for $1 \leq i < N_{\mathcal{I}}$, and $\lambda_{N_{\mathcal{I}}} = 1 - p_{\mathrm{d},N_{\mathcal{I}}}$ for the final class. The corresponding eigenvectors can be found analytically through straightforward forward-substitution.

The population eventually converges to a stationary ecological-like equilibrium, $\mathbf{x}^*$. Mathematically, this corresponds exactly to the normalized principal eigenvector of the transition matrix $\mathbf{A}$ (associated with the largest eigenvalue). Because probabilities of death and transition monotonically increase with instability, the maximum eigenvalue is definitively the first one:
$$\lambda_{\max} = \lambda_1 = 1 - p_{\mathrm{d},1} - p_{\mu,1}$$
To find the stationary subpopulation concentrations, we solve the eigenvector system $\mathbf{A} \mathbf{x}^* = \lambda_1 \mathbf{x}^*$ recursively using forward substitution. For classes $1 < i \leq N_{\mathcal{I}}$, this yields:
$$x_i^* = x_{i-1}^* \frac{p_{\mu, i-1}}{\lambda_1 - \lambda_i}$$
By substituting the eigenvalues and expanding the recursion, we obtain a complete product formula relative to the first class:
$$x_i^* = x_1^* \prod_{j=2}^i \frac{p_{\mu, j-1}}{(p_{\mathrm{d},j} + p_{\mu,j}) - (p_{\mathrm{d},1} + p_{\mu,1})}$$
Finally, we apply the biological constraint that the sum of all subpopulation fractions must equal 1 ($\sum_{i=1}^{N_{\mathcal{I}}} x_i^* = 1$). This provides the absolute fraction for the lowest instability class:
$$x_1^* = \left( 1 + \sum_{i=2}^{N_{\mathcal{I}}} \prod_{j=2}^i \frac{p_{\mu, j-1}}{(p_{\mathrm{d},j} + p_{\mu,j}) - (p_{\mathrm{d},1} + p_{\mu,1})} \right)^{-1}$$