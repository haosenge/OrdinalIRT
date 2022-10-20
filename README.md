# Gibbs Sampler for Ordinal IRT Models

This repo provides an efficient Gibbs Sampler for dynamic item response theory models with ordinal outcomes. An application can be found in 
[Haosen Ge (2021) "Measuring Regulatory Barriers Using Annual Reports of Firms"](https://www.haosenge.net/_files/ugd/557840_20bf6709b255488d8466ae0d59c47716.pdf).

The model assumes the followiing data generating process:

$$U_{ijt} = \begin{cases}
		1 \quad \text{if} \quad U_{ijt}^* \leq \alpha_{ij}^R \\
		2  \quad \text{if}\quad  \alpha_{ij}^R  < U_{ijt}^* \leq  \alpha_{ij}^E \\
		3  \quad \text{if} \quad  U_{ijt}^* > \alpha_{ij}^E
	\end{cases}$$
  
Denote the set of $\{\theta_{j,t}\}$ as 
$\Theta$ 
and the set of 
$\\{\alpha_{ij}^E\\}$ 
and 
$\\{\alpha_{ij}^R\\}$ 
as 
$\alpha^E$ 
and 
$\alpha^R$. 
Let 
$U$ 
denote the observed data and 
$U^*$ 
the augmented data. We can write the full data likelihood with the augmented data as:

$$
		\mathcal{L}(\Theta, \alpha^E, \alpha^R | U, U^* ) = 
    \prod_{t=1}^T \prod_{j = 1}^J \prod_{i = 1}^I 
    \\{ I(U_{ijt} = 1,U_{ijt}^* \leq \alpha_{ij}^R) + 
    I(U_{ijt} = 2,  \alpha_{ij}^R  < U_{ijt}^* \leq  \alpha_{ij}^E) + 
    I(U_{ijt} = 3, U_{ijt}^* > \alpha_{ij}^E)\\} \cdot \phi_{\theta_{jt}}(U_{ijt}^*)
$$

where 
$\phi_{\theta_{jt}(\cdot)}$ denotes the probability density function of $\mathcal{N}(\theta_{jt}, 1)$.

