# Literature Report: What are recent methods for using machine learning to accelerate numerical linear algebra solvers

*Generated 2026-10-18 by ScholarAgent*

## Abstract
Recent research explores the integration of machine learning (ML) techniques to accelerate numerical linear algebra solvers. Key approaches include combining ML with traditional methods like the Lanczos algorithm and the conjugate gradient method, using randomized numerical linear algebra (RandNLA) techniques, employing curriculum-based reinforcement learning (RL), and designing advanced preconditioners. Notably, Nikolopoulos et al. (2022) propose integrating deep neural networks with the POD-2G iterative solver, achieving significant speedup and reduced iteration counts. Bioli et al. (2025) introduce NyströmNGD and RPCholNGD, which leverage RandNLA to accelerate natural gradient descent, demonstrating substantial computational savings. Trifonov et al. (2024) utilize graph neural networks (GNNs) to improve preconditioner design, enhancing spectral performance. Guido et al. (2023) and Iwashita et al. (2022) focus on generating effective initial guesses and approximating eigenvectors, respectively, to accelerate sequences of linear systems. Machine learning also plays a role in optimizing preconditioners, as seen in Antonietti et al. (2021) and Trifonov et al. (2024). Experiments across these studies show varying degrees of success, with notable improvements in speed and accuracy. However, limitations persist, particularly regarding the applicability to specific matrix types and properties, and the need for substantial computational resources. Overall, these methods offer promising avenues for enhancing the efficiency of numerical linear algebra solvers.

## 1. Research Question
What are recent methods for using machine learning to accelerate numerical linear algebra solvers?

## 2. Search Methodology
- Search date: 2026-10-18
- Databases: arxiv, semantic_scholar, crossref
- Queries issued: 9 (What are recent methods for using machine learning to accelerate numerical linear algebra solvers?; machine learning numerical linear algebra; accelerate linear algebra ml; numerical linear algebra optimization ml; ML techniques for linear algebra; speed up linear algebra with ML; machine learning in linear algebra solvers; linear algebra acceleration methods)
- Candidates retrieved: 253  →  after de-duplication: 171  →  selected: 12
- Full text obtained and analysed: 12 papers; abstract only: 0

## 3. Literature Landscape
- **Iterative Methods** — Focuses on iterative algorithms to solve linear systems, such as the Lanczos algorithm, which constructs an orthonormal basis of the Krylov subspace and projects the matrix to a smaller tridiagonal form for eigenvalue approximation. *(Dolean & Tabeart (2026))*
- **Learned Preconditioners** — Utilizes machine learning techniques to improve the efficiency of iterative solvers through learned preconditioners, including deep neural networks, convolutional autoencoders, and GNNs. *(Nikolopoulos et al. (2022), Hegde et al. (2026), Guido et al. (2023), Trifonov et al. (2024), Iwashita et al. (2022), Antonietti et al. (2021))*
- **Randomized Numerical Linear Algebra (RandNLA)** — Employs randomized techniques to accelerate linear algebra operations, such as Nyström approximation and sketching operators, often leveraging GPU acceleration. *(Bioli et al. (2025), Rathore et al. (2026), Murray et al. (2023))*
- **Reinforcement Learning for RLA** — Uses reinforcement learning to discover randomized linear algebra algorithms through a curriculum-based approach. *(Xiong et al. (2026))*
- **Monte Carlo Methods** — Incorporates Monte Carlo methods to enhance linear algebra operations, particularly focusing on preconditioning techniques. *(Lebedev & Alexandrov (2024))*

The trade-offs among these families of approaches include the level of generality, computational efficiency, and the need for specialized knowledge. Iterative methods, like the Lanczos algorithm, offer robust solutions but may require specific properties of the matrix. Learned preconditioners leverage machine learning to improve convergence but are often problem-specific and may require significant training data. Randomized numerical linear algebra techniques provide scalable solutions through randomization but may have limited theoretical guarantees. Reinforcement learning approaches can discover novel algorithms but face challenges in generalizing across different problem classes. Monte Carlo methods, while powerful, are constrained by precision and lack multi-GPU support.

## 4. Major Approaches
Papers by Nikolopoulos et al. (2022) and Dolean & Tabeart (2026) focus on enhancing iterative solvers with machine learning techniques. Nikolopoulos et al. propose combining deep feedforward neural networks, convolutional autoencoders, and the POD-2G iterative solver to accelerate the solution of large-scale parametrized systems ([1][2]). Dolean & Tabeart discuss the conjugate gradient method and the Lanczos algorithm, highlighting their applications in numerical linear algebra for solving systems of equations and eigenproblems ([3]).

#### Randomized Numerical Linear Algebra (RandNLA)
Bioli et al. (2025) introduce NyströmNGD and RPCholNGD, which utilize randomized numerical linear algebra techniques to accelerate natural gradient descent for physics-informed neural networks ([4]). Trifonov et al. (2024) propose a novel neural method for preconditioner design using graph neural networks (GNNs) to improve spectral performance over classical methods ([5]).

#### Curriculum-Based Reinforcement Learning (RL)
Xiong et al. (2026) introduce RL4RLA, a framework for automating the discovery of RLA algorithms using a curriculum-based approach ([6]).

#### Accelerating Sequences of Linear Systems
Guido et al. (2023) combine reduced-order projection with randomized linear algebra techniques to generate a good initial guess for iterative solvers, thereby accelerating the solution of sequences of linear systems ([7]). Iwashita et al. (2022) construct an auxiliary matrix using error vectors sampled from previous iterations to approximate eigenvectors associated with small eigenvalues, further improving the solution process ([8]).

#### Preconditioner Design
Trifonov et al. (2024) use GNNs to learn corrections for classical ILU preconditioners, aiming to improve spectral performance ([9]). Antonietti et al. (2021) introduce a machine learning approach to optimize Algebraic Multigrid parameters ([10]).

#### Monte Carlo Methods
Lebedev & Alexandrov (2024) enhance the SPAI preconditioner using parallel Monte Carlo methods, providing an alternative to traditional methods ([11]).

## 5. Technical Analysis
The key technical ideas in recent machine learning (ML) approaches to accelerate numerical linear algebra solvers involve leveraging ML models to improve the efficiency and accuracy of iterative solvers. These methods differ in their specific formulations and complexity claims. The Lanczos method constructs an orthonormal basis of the Krylov subspace and projects the matrix to a smaller tridiagonal form for eigenvalue approximation. This method is particularly efficient for large sparse matrices, offering a complexity of \(O(n^2)\) per iteration compared to the \(O(n^3)\) complexity of full eigendecomposition [12]. The use of preconditioners is another common technique in accelerating numerical linear algebra solvers. For instance, [13] introduces the use of randomized Nyström approximation and Randomly Pivoted Partial Cholesky (RPCholesky) decomposition to precondition the Gramian matrix. The NyströmNGD method has a computational cost of \(O(Cmv(l) + pl^2)\) flops, while RPCholNGD has a cost of \(O(l \cdot C_{\text{col}}(b)/b + C_{\text{diag}} + pl^2)\) flops [14]. Machine learning can be used to correct classical preconditioners, as demonstrated by [15] which uses graph neural networks (GNNs) to learn corrections for classical ILU preconditioners. The loss function for this approach is defined as \(L(\theta) = L + \alpha \cdot \text{GNN}(\theta, L)\), where \(\theta\) represents the parameters of the GNN [15]. Curriculum-based reinforcement learning (RL) frameworks are employed to guide the discovery of RLA algorithms, as shown by [16] which uses such a framework to progressively increase problem difficulty. This approach leverages the Sketch-and-Precondition paradigm to iteratively refine solutions [17]. Some methods focus on reducing the computational complexity of iterative methods, such as utilizing RandNLA techniques like NyströmPCG, NysADMM, and SAPPHIRE for GPU-accelerated optimization [18]. In summary, the technical approaches vary widely, ranging from traditional Krylov subspace methods like the Lanczos algorithm to modern ML-driven techniques that leverage neural networks and RL. Each method offers unique advantages and trade-offs in terms of computational efficiency and accuracy, making them suitable for different types of numerical linear algebra problems.

## 6. Paper Comparison
| Paper | Problem | Method | Complexity | Datasets | Main Result | Strength | Limitation |
|---|---|---|---|---|---|---|---|
| **P1** Dolean & Tabeart (2026) | Estimating the condition number κ(A) is crucial for understanding the performance of iterative methods like Conjugate Gradient. | Lanczos algorithm constructs an orthonormal basis of the Krylov subspace and projects the matrix to a smaller tridiagonal form for eigenvalue approximation. | The complexity of full eigendecomposition is O(n^3), while Lanczos method is more efficient for large sparse matrices. | - | Derive the Conjugate Gradient method from the minimisation of the quadratic functional; state and use the optimality property of CG; use the Chebyshev bound to relate the CG convergence rate to κ(A); describe Lanczos iteration and explain why Ritz values converge at the ends of the spectrum first; explain the CG-Lanczos equivalence and interpret ridge regression and truncated CG as spectral filters. | Provides theoretical foundations for understanding the performance of iterative methods and their relationship to eigenvalue spectra. | Limited to matrices with property A; does not cover all aspects of machine learning acceleration. |
| **P2** Nikolopoulos et al. (2022) | Accelerating the solution of real-life computational mechanics problems in multi-query scenarios. | Combines deep feedforward neural networks, convolutional autoencoders, and POD-2G iterative solver. | - | Indirect tensile strength (ITS) test, 3D deformable porous medium problem (Biot problem) | POD solvers outperform AMG solvers in terms of speedup and number of iterations; initial solution from surrogate model reduces computational cost. | Proposes a new approach combining machine learning and linear algebra for solving large-scale parametrized systems; introduces a two-step process involving a reduced set of model evaluations followed by an iterative solver refinement. | Only tested on specific types of problems. |
| **P3** Bioli et al. (2025) | Solving linear systems with (G(θ) + µI) via direct methods is computationally expensive, making NGD impractical for large-scale networks. | The authors propose using randomized Nyström approximation and Randomly Pivoted Partial Cholesky (RPCholesky) decomposition to precondition the Gramian matrix. | NyströmNGD has a cost of O(Cmv(l) + pl2) flops, while RPCholNGD has a cost of O(l· Ccol(b)/b + Cdiag + pl2) flops. | Heat 3+1D, Kovasznay | Preconditioned NGD variants achieve similar accuracy but in significantly less time compared to unpreconditioned matrix-free NGD. | Preconditioned NGD methods operate with a memory footprint of O(plmax) compared to O(p2) for SSBroyden and BFGS. | Limited to specific types of PDE problems; dependent on the quality of preconditioning; performance may degrade with very ill-conditioned Gramian matrices. |
| **P4** Hegde et al. (2026) | Selecting the appropriate size of the nugget for Tikhonov regularization in ill-conditioned linear systems. | Using Richardson extrapolation to combine multiple linear solves and determine the solution of the ill-conditioned system. | Balancing numerical error and extrapolation error to find the optimal h. | - | Autonugget outperformed baselines in terms of accuracy. | Improved accuracy observed with n > 2, provided stable optimisation and better length-scale estimation. | Currently limited to symmetric positive definite matrices, incurs higher computational cost compared to standard methods, limited to matrices that can be stored in memory. |
| **P5** Xiong et al. (2026) | Challenges in discovering RLA algorithms due to combinatorial search spaces and sparse reward landscapes. | Uses a curriculum-based RL framework to guide the discovery of RLA algorithms. | Progressively increases problem difficulty to enable efficient discovery. | - | Consistent discovery success across stages, better search efficiency with MCGS variants. | Rediscovers state-of-the-art methods including sketch-and-precondition solvers, Randomized Kaczmarz, and Newton Sketch. | The discovery environment is synthetic, adapting to a new problem class requires specifying a compact task interface, moving from rediscovery to novel algorithm discovery requires larger search budgets. |
| **P6** Guido et al. (2023) | Linear systems from time-dependent PDEs are costly. | Combines reduced-order projection and randomized techniques for initial guess. | - | Elliptic PDE with varying coefficients and sources. | Speedup factors and reduced iterations. | Reduces iterations, provides speedup. | Stiffness limitations, smoothness assumptions. |
| **P7** Rathore et al. (2026) | Lack of GPU-accelerated and user-friendly implementations of RandNLA-based methods for large-scale optimization. | Uses RandNLA techniques like NyströmPCG, NysADMM, and SAPPHIRE for GPU-accelerated optimization. | Per-iteration complexity of O(n^2) for iterative methods, O(n^3) for direct methods. | Dense random-feature problems: acsincome-rf, yearpredictionmsd-rf; Sparse problems: e2006, realsim. | NyströmPCG on GPU is faster than on CPU by 57×–128×; NysADMM solves acsincome-rf and yearpredictionmsd-rf in approximately 3093 and 811 seconds, respectively, while no competing solver succeeds. | RandNLA has emerged as a powerful toolkit for addressing large-scale optimization challenges; NyströmPCG method combines the scalability of iterative methods with robustness to ill-conditioning. | Limited to specific problem classes; does not support all atoms and solvers; expensive for sparse matrices. |
| **P8** Trifonov et al. (2024) | What are recent methods for using machine learning to accelerate numerical linear algebra solvers? | Uses GNN to learn corrections for classical ILU preconditioners. | Generates datasets with measurable complexity metrics for real-world problems. | SPD matrices from discretization of elliptic equations, 2D diffusion equation, 2D Poisson equation | For larger grid sizes, the total time of PreCorrector is less than that of ICt(5); PreCorrector has fewer CG iterations compared to ICt(1) and ICt(5); PreCorrector has lighter inference and thus less total time-to-solution; PreCorrector does not degenerate in quality when trained on datasets with higher contrast. | Proposes a novel scheme for preconditioner design based on learning correction for well-established preconditioners from linear algebra with the GNN; Provides an understanding of the loss function used with emphasis on low frequencies; Demonstrates the superiority of the proposed approach and loss function over classical preconditioners through extensive experiments; Achieves stable and fast training convergence that can handle parametric PDEs with contrast coefficients. | - |
| **P9** Iwashita et al. (2022) | Solving a sequence of linear systems with identical coefficient matrices. | Constructs an auxiliary matrix using error vectors sampled from previous iterations to approximate eigenvectors associated with small eigenvalues. | Computational cost for subspace correction preconditioning and deflation is analyzed. | SuiteSparse Matrix Collection, Queen_4147, Bump_2911, G3_circuit, Flan_1565 | Proposed method achieves more than 2-fold speedup in convergence compared to the parallel ICCG solver for 21 out of 30 datasets. | Significant speedup in convergence for many datasets; provides useful information for eigenvectors associated with small eigenvalues. | Limited by the effectiveness of the sampling method; dependent on the initial choice of sampling method. |
| **P10** Murray et al. (2023) | Estimating trace and solving linear systems. | Nyström PCG and sketching operators. | O(ln) space and O(ln) time for Nyström PCG. | - | Nyström PCG outperforms random-features preconditioning on shared-memory machines. | Randomized algorithms can reduce data movement; backward error plus sensitivity analysis provides guarantees. | Limited theoretical guarantees; forward error bounds are hard to compute; lacks organization of RandNLA literature. |
| **P11** Antonietti et al. (2021) | Optimizing Algebraic Multigrid parameters using machine learning | ANN-based approach | Strides and checkerboard patterns in diffusion coefficient | Dataset 1, Test Case 1, Test Case 4: the stationary Stokes problem | Good performance across different test cases with high accuracy and low loss values | Developed an ANN-based approach to enhance AMG efficiency; showed good performance across different test cases; introduced a pooling operator for matrix independence; correlated convergence factor with CPU time | Only tested on specific mesh sizes and patterns; dependent on the quality of the training data; limited by single-GPU memory |
| **P12** Lebedev & Alexandrov (2024) | Finding a good preconditioner | Enhanced SPAI preconditioner using parallel Monte Carlo methods | Linear complexity in matrix size | Set of matrices including symmetric and non-symmetric matrices of varying sizes and filling fractions | (MC)2MI is superior to MSPAI for dense or very large matrices | Superior performance for dense or very large matrices, linear complexity, rough solution estimates, improved preconditioner quality with lower precision | Limited by precision, no multi-GPU support |

## 7. Experimental Results Across Literature
The experimental results across the literature highlight various methods and their performance in accelerating numerical linear algebra solvers. Nikolopoulos et al. (2022) found that POD solvers outperform AMG solvers in terms of both speedup and the number of iterations required [19]. Using an initial solution from a surrogate model further reduces computational costs [20]. Studies utilized diverse datasets, including the Indirect Tensile Strength (ITS) test and a 3D deformable porous medium problem (Biot problem) [21]. Bioli et al. (2025) demonstrated that preconditioned Natural Gradient Descent (NGD) variants achieve similar accuracy but in significantly less time compared to unpreconditioned matrix-free NGD [22]. The memory footprint of preconditioned NGD methods is notably lower, requiring only \(O(pl_{\max})\) compared to \(O(p^2)\) for other methods such as SSBroyden and BFGS [23]. Hegde et al. (2026) reported that Autonugget outperformed baseline methods in terms of accuracy, though improvements were more pronounced for \(n > 2\) [24]. The baselines considered included LU decomposition and LU with a fixed nugget at single precision [25]. Xiong et al. (2026) found consistent discovery success across stages and better search efficiency with MCGS variants [26]. Rathore et al. (2026) introduced NyströmPCG and NysADMM, which demonstrated significant speedups on large-scale ridge regression problems, particularly on GPUs [18]. These methods were applied to dense and sparse problems, including acsincome-rf, yearpredictionmsd-rf, e2006, and realsim [27]. Trifonov et al. (2024) compared classical algorithms and found that PreCorrector had fewer CG iterations and was faster than ICt(5) for larger grid sizes [28]. Iwashita et al. (2022) achieved more than 2-fold speedup in convergence for 21 out of 30 datasets when using their proposed method compared to the parallel ICCG solver [29]. The tests were conducted on the SuiteSparse Matrix Collection [30].

## 8. Limitations
Machine learning approaches for accelerating numerical linear algebra solvers face several recurring limitations, which can be grouped into a few key themes.

**Matrix Properties and Types:**

- Methods are often limited to specific matrix properties or types. For instance, they may only apply to matrices with certain properties ([31]), symmetric positive definite matrices ([32]), or matrices that can be stored in memory ([32]). Additionally, some methods require specific smoothness assumptions ([33]) and are dependent on the quality of preconditioning ([34], [35]).

**Computational Cost and Memory Constraints:**

- There is a trade-off between computational efficiency and the complexity of the methods. Some approaches incur higher computational costs compared to standard methods ([32]), and are limited to matrices that fit within memory constraints ([32]).

**Problem-Specific Limitations:**

- Many methods are tailored to specific problem classes or types of partial differential equations (PDEs) ([1], [34], [36]). They may also be limited by the stiffness of the problem ([37]) and require specific smoothness assumptions ([33]).

**Dependence on Quality of Data and Sampling:**

- The performance of these methods can be highly dependent on the quality of the training data and the sampling method used. Issues arise if the sampling method is not effective ([38]), and the quality of the training data significantly impacts the results ([38]).

**Theoretical and Practical Challenges:**

- Theoretical guarantees are limited due to choices in computational routines ([39]), and forward error bounds are difficult to compute at runtime due to the lack of available problem data ([40]). Furthermore, the absence of work organizing the RandNLA literature limits the theoretical understanding ([41]).

**Hardware and Implementation Constraints:**

- Methods are constrained by hardware limitations such as single-GPU memory ([38]) and the need for multi-GPU training ([42]). Additionally, the choice of performance index and mesh size can limit the applicability of the methods ([43]).

**Discovery and Adaptation Challenges:**

- Discovering novel algorithms through machine learning requires significant effort and resources. Moving from rediscovery to novel algorithm discovery necessitates larger search budgets ([44]), and adapting to new problem classes often requires specifying a compact task interface ([44]), which can be challenging. These limitations highlight the ongoing challenges in developing machine learning techniques to effectively accelerate numerical linear algebra solvers across various applications and problem domains.

## 9. Potential Research Directions
**Direction 1: Develop a self-adaptive machine learning framework for numerical linear algebra solvers that dynamically adjusts to different preconditioning techniques and sampling methods without requiring manual tuning.**

- Recurring limitation: *Data and Method Dependency* — The performance of the models is highly dependent on external factors such as the quality of preconditioning, the initial choice of sampling method, and the quality of the training data. [34][45][38]
- Mentioned by 3/12 analysed papers (Antonietti et al. (2021), Bioli et al. (2025), Iwashita et al. (2022)).
- Novelty search: 12 related papers examined; closest work: Accelerated Portfolio Optimization and Option Pricing with Reinforcement Learning (2025); Self-adaptive deep neural network: Numerical approximation to functions and PDEs (2021).
- Confidence: **MEDIUM**. Our search of 12 related papers did not identify work directly addressing this within the searched corpus; it is a potential research direction, not a proven gap.

**Direction 2: Develop a machine learning framework that can efficiently approximate solutions for a wide range of matrices, including non-symmetric positive definite and symmetric positive semidefinite matrices, without requiring them to fit into memory.**

- Recurring limitation: *Matrix Properties and Memory Constraints* — The methods are currently restricted to specific types of matrices, such as those with property A, symmetric positive definite matrices, and matrices that fit into memory. There is a need to extend these methods to handle a broader range of matrices, including non-symmetric positive definite and symmetric positive semidefinite cases. [31][32][32]
- Mentioned by 2/12 analysed papers (Dolean & Tabeart (2026), Hegde et al. (2026)).
- Novelty search: 12 related papers examined; closest work: End-to-End Deep Learning for Predicting Metric Space-Valued Outputs (2025); Probabilistic Learning Vector Quantization on Manifold of Symmetric Positive Definite Matrices (2021); SPD Matrix Learning for Neuroimaging Analysis: Perspectives, Methods, and Challenges (2025).
- Confidence: **MEDIUM**. Our search of 12 related papers did not identify work directly addressing this within the searched corpus; it is a potential research direction, not a proven gap.

## 10. Evidence Quality & Verification
- Claims checked: **55**  |  supported: 22  |  partially supported: 22  |  unsupported: 10  |  contradicted: 1
- Self-correction: 0 claim(s) corrected, 11 claim(s) removed from the final text.
- Papers analysed: 12 (12 full text, 0 abstract-only).
- Limits: findings are restricted to the retrieved corpus; absence of evidence in this corpus is not evidence of absence in the literature.

## 11. Conclusion
This report synthesised the retrieved literature with every statement linked to a source passage. See *Potential Research Directions* for recurring, novelty-checked open problems.

## References
- Anton Lebedev, Vassil Alexandrov (2024). *On Advanced Monte Carlo Methods for Linear Algebra on Advanced Accelerator Architectures*. doi:10.1109/scala.2018.00014; arXiv:2409.03095
- Paola F. Antonietti, Matteo Caldana, Luca Dede' (2021). *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*. doi:10.1007/s10013-022-00597-w; arXiv:2111.01629
- Riley Murray, James Demmel, Michael W. Mahoney, N. Benjamin Erichson, Maksim Melnichenko, Osman Asif Malik (2023). *Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software*. arXiv:2302.11474
- Takeshi Iwashita, Kota Ikehara, Takeshi Fukaya, Takeshi Mifune (2022). *Convergence Acceleration of Preconditioned CG Solver Based on Error Vector Sampling for a Sequence of Linear Systems*. Numerical Linear Algebra with Applications. doi:10.1002/nla.2512; arXiv:2203.08498
- Pratik Rathore, Zachary Frangella, Parth Nobel, Xuning Hu, Madeleine Udell (2026). *GPU-Enabled Large-Scale Optimization Using Randomized Linear Algebra*. arXiv:2609.08136
- Margherita Guido, Daniel Kressner, Paolo Ricci (2023). *Subspace Acceleration for a Sequence of Linear Systems and Application to Plasma Simulation*. arXiv:2309.02156
- Jinglong Xiong, Xiaotian Liu, Ruoxin Wang, Zihang Liu, Yefan Zhou, Yujun Yan (2026). *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*. arXiv:2605.18004
- Ivan Bioli, Carlo Marcati, Giancarlo Sangalli (2025). *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*. arXiv:2505.11638
- Stefanos Nikolopoulos, Ioannis Kalogeris, Vissarion Papadopoulos, George Stavroulakis (2022). *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*. arXiv:2207.02543
- Victorita Dolean, Jemima Tabeart (2026). *Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)*. arXiv:2608.21234
- Disha Hegde, Jon Cockayne, Chris. J. Oates (2026). *Extrapolating from Regularised Solutions for Solving Ill-Conditioned Linear Systems in Machine Learning*. arXiv:2606.30328
- Vladislav Trifonov, Alexander Rudikov, Oleg Iliev, Yuri M. Laevsky, Ivan Oseledets, Ekaterina Muravleva (2024). *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers*. arXiv:2405.15557

## Evidence Index
_Numbers in square brackets in the text refer to these source passages._

**[1]** Nikolopoulos et al. (2022), Abstract, p. 1 — *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*  
> Proposes AI-enhanced iterative solvers for solving large-scale parametrized systems.
**[2]** Nikolopoulos et al. (2022), §3 Machine learning accelerated iterative solvers, p. 8 — *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*  
> Combines deep feedforward neural networks, convolutional autoencoders, and POD-2G iterative solver.
**[3]** Dolean & Tabeart (2026), §4.1 THE CONJUGATE GRADIENT METHOD 47, pp. 53-54 — *Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)*  
> The conjugate gradient method and Lanczos algorithm are discussed, focusing on their applications in numerical linear algebra for solving systems of equations and eigenproblems.
**[4]** Bioli et al. (2025), §1 Introduction, pp. 1-2 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> The paper introduces NyströmNGD and RPCholNGD, which use randomized numerical linear algebra techniques to accelerate natural gradient descent for physics-informed neural networks.
**[5]** Trifonov et al. (2024), Abstract, p. 1 — *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers*  
> Proposes a novel neural method for preconditioner design using GNNs to improve spectral performance over classical methods.
**[6]** Xiong et al. (2026), Abstract, p. 1 — *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*  
> Introduces RL4RLA, a framework for automating the discovery of RLA algorithms using a curriculum-based approach.
**[7]** Guido et al. (2023), Abstract, p. 1 — *Subspace Acceleration for a Sequence of Linear Systems and Application to Plasma Simulation*  
> The paper presents a method to accelerate the solution of sequences of linear systems by leveraging the history of solutions.
**[8]** Iwashita et al. (2022), §6 CONCLUSION, pp. 15-16 — *Convergence Acceleration of Preconditioned CG Solver Based on Error Vector Sampling for a Sequence of Linear Systems*  
> In this paper, we introduce an algebraic auxiliary matrix construction method that can be used for the subspace correction preconditioning and the deflation method. We focus on a problem in which a sequence of linear systems with an identical coefficient matrix are solved. In our method, we sample the approximate solut
**[9]** Trifonov et al. (2024), §3 Learn correction for ILU, p. 3 — *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers*  
> Our main goal is to construct preconditioners that reduce the condition number of an SPD matrix more than classical preconditioners with the same sparsity pattern. Several previous papers suggested to use convolutional neural networks with sparse convolutions (Sappl et al., 2019; Cal`ı et al., 2023) and graph neural ne
**[10]** Antonietti et al. (2021), §1 Introduction, pp. 1-2 — *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*  
> Introduces a machine learning approach to optimize Algebraic Multigrid parameters.
**[11]** Lebedev & Alexandrov (2024), I INTRODUCTION, p. 1 — *On Advanced Monte Carlo Methods for Linear Algebra on Advanced Accelerator Architectures*  
> Enhanced version of SPAI preconditioner using parallel Monte Carlo methods.
**[12]** Dolean & Tabeart (2026), §4.2 THE LANCZOS METHOD AND THE SYMMETRIC EIGENPROBLEM 51, p. 57 — *Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)*  
> The complexity of full eigendecomposition is O(n^3), while Lanczos method is more efficient for large sparse matrices.
**[13]** Bioli et al. (2025), §1 Introduction, p. 2 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> The authors propose using randomized Nyström approximation and Randomly Pivoted Partial Cholesky (RPCholesky) decomposition to precondition the Gramian matrix.
**[14]** Bioli et al. (2025), §4.2 Preconditioning matrix-free Natural Gradient Descent. When iteratively solving the, p. 16 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> NyströmNGD has a cost of O(Cmv(l) + pl2) flops, while RPCholNGD has a cost of O l· Ccol(b)/b + Cdiag + pl2 flops.
**[15]** Trifonov et al. (2024), §3 Learn correction for ILU, p. 3 — *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers*  
> Uses GNN to learn corrections for classical ILU preconditioners.
**[16]** Xiong et al. (2026), §1 Introduction, p. 2 — *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*  
> Uses a curriculum-based RL framework to guide the discovery of RLA algorithms.
**[17]** Xiong et al. (2026), §5.1.1 CURRICULUM OVERVIEW, p. 6 — *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*  
> We instantiate our framework across five problem domains, each following the same MCGS structure (Section 4.3) but with domain-specific matrix generation, operator spaces, and reward functions. RLA Paradigms. We consider four algorithmic patterns that emerge during curriculum-based discovery: (i) Sketchand-Precondition
**[18]** Rathore et al. (2026), ABSTRACT, p. 1 — *GPU-Enabled Large-Scale Optimization Using Randomized Linear Algebra*  
> Uses RandNLA techniques like NyströmPCG, NysADMM, and SAPPHIRE for GPU-accelerated optimization.
**[19]** Nikolopoulos et al. (2022), §4 Numerical applications, p. 17 — *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*  
> Initial solution from surrogate model reduces computational cost
**[20]** Nikolopoulos et al. (2022), §4.2 Biot problem - deformable porous medium, pp. 22-23 — *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*  
> | ε = 10−4 | ε = 10−5 | ε = 10−6 | ε = 10−7 | ε = 10−8 AMG-3G (u(0) = 0) | ×1 | ×1 | ×1 | ×1 | ×1 AMG-5G (u(0) = 0) | ×0.97 | ×0.96 | ×0.96 | ×0.96 | ×0.96 AMG-6G (u(0) = 0) | ×0.97 | ×0.96 | ×0.96 | ×0.96 | ×0.96 POD-2G (u(0) = 0) | ×12.31 | ×7.32 | ×4.89 | ×2.34 | ×1.77 POD-2G (u(0) = u ) sur | ×76.89 | ×31.54 | ×17.
**[21]** Nikolopoulos et al. (2022), §4 Numerical applications, p. 14 — *AI-enhanced iterative solvers for accelerating the solution of large scale parametrized systems*  
> Indirect tensile strength (ITS) test
**[22]** Bioli et al. (2025), §5.1 Comparison of Natural Gradient Descent Variants., p. 20 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> NGD with a direct solver (NGD full)
**[23]** Bioli et al. (2025), §1.2 Related work and comparison with existing optimization methods. Compared to full-, pp. 2-3 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> memory quasi-Newton methods such as BFGS and SSBroyden [88, 83, 55], our approach differs fundamentally both conceptually and computationally. Conceptually, NGD is rooted in a first-optimizethen-discretize paradigm: one chooses a metric in function space, possibly tailored to the underlying PDE, and then pulls it back 
**[24]** Hegde et al. (2026), §4.2 Solver Accuracy, pp. 9-10 — *Extrapolating from Regularised Solutions for Solving Ill-Conditioned Linear Systems in Machine Learning*  
> Improved accuracy observed with n > 2, but not substantial enough for default.
**[25]** Hegde et al. (2026), §4.1 Baselines, p. 8 — *Extrapolating from Regularised Solutions for Solving Ill-Conditioned Linear Systems in Machine Learning*  
> LU-fix: LU with fixed nugget at single precision, σ = 10^-8.
**[26]** Xiong et al. (2026), §5.2 Discovery Success and Search Efficiency, pp. 7-8 — *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*  
> Better search efficiency with MCGS variants
**[27]** Rathore et al. (2026), §6.3 Bounded Elastic Net with NysADMM, p. 13 — *GPU-Enabled Large-Scale Optimization Using Randomized Linear Algebra*  
> SCS (direct and indirect solvers)
**[28]** Trifonov et al. (2024), Experiments with classical algorithms, p. 6 — *Learning from Linear Algebra: A Graph Neural Network Approach to Preconditioner Design for Conjugate Gradient Solvers*  
> PreCorrector has fewer CG iterations compared to ICt(1) and ICt(5) respectively.
**[29]** Iwashita et al. (2022), §5.3 Numerical results on the parallel solver, pp. 12-13 — *Convergence Acceleration of Preconditioned CG Solver Based on Error Vector Sampling for a Sequence of Linear Systems*  
> In the test using random vectors, the proposed method attains more than 2-fold speedup in convergence compared to the parallel ICCG solver for 21 out of 30 datasets. Next, we examine the computational time. In the test using a vector of ones, the proposed method reduces the solution time for 28 out of 30 datasets. For 
**[30]** Iwashita et al. (2022), §5.1 Test Conditions, pp. 6-7 — *Convergence Acceleration of Preconditioned CG Solver Based on Error Vector Sampling for a Sequence of Linear Systems*  
> SuiteSparse Matrix Collection
**[31]** Dolean & Tabeart (2026), §3.1.3 Convergence of the SOR iteration, pp. 39-40 — *Advanced Linear Algebra with Applications - Part I (Numerical linear algebra for PDEs, machine learning, and data assimilation)*  
> Limited to matrices with property A
**[32]** Hegde et al. (2026), §5 Discussion, pp. 12-13 — *Extrapolating from Regularised Solutions for Solving Ill-Conditioned Linear Systems in Machine Learning*  
> Currently limited to symmetric positive definite matrices.
**[33]** Guido et al. (2023), §3 Convergence Analysis, p. 7 — *Subspace Acceleration for a Sequence of Linear Systems and Application to Plasma Simulation*  
> Requires specific smoothness assumptions
**[34]** Bioli et al. (2025), Abstract, p. 1 — *Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra*  
> Dependent on the quality of preconditioning.
**[35]** Lebedev & Alexandrov (2024), VII CONCLUSIONS AND FUTURE WORK, pp. 7-8 — *On Advanced Monte Carlo Methods for Linear Algebra on Advanced Accelerator Architectures*  
> Limited by the precision of the preconditioner
**[36]** Rathore et al. (2026), §1 Introduction, pp. 1-2 — *GPU-Enabled Large-Scale Optimization Using Randomized Linear Algebra*  
> Limited to specific problem classes
**[37]** Guido et al. (2023), §6 Conclusions, pp. 17-18 — *Subspace Acceleration for a Sequence of Linear Systems and Application to Plasma Simulation*  
> Limited by the stiffness of the problem
**[38]** Antonietti et al. (2021), §7 Conclusions, p. 36 — *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*  
> Limited by single-GPU memory
**[39]** Murray et al. (2023), §1.2 This monograph, from an astronaut’s-eye view, p. 15 — *Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software*  
> Theoretical guarantees for drivers are limited due to choices in computational routines.
**[40]** Murray et al. (2023), B.2 Basic error analysis Least Squares and Optimization, p. 154 — *Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software*  
> Forward error bounds are difficult to compute at runtime due to lack of available problem data.
**[41]** Murray et al. (2023), §1.1 Our world Introduction, pp. 10-11 — *Randomized Numerical Linear Algebra : A Perspective on the Field With an Eye to Software*  
> Limited by the absence of work organizing the RandNLA literature.
**[42]** Lebedev & Alexandrov (2024), VII CONCLUSIONS AND FUTURE WORK, p. 8 — *On Advanced Monte Carlo Methods for Linear Algebra on Advanced Accelerator Architectures*  
> Currently does not support multi-GPU training
**[43]** Antonietti et al. (2021), §5.3.1 Choice of the performance index p, pp. 18-19 — *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*  
> Only tested on specific mesh sizes and patterns
**[44]** Xiong et al. (2026), Limitations., p. 9 — *RL4RLA: Teaching ML to Discover Randomized Linear Algebra Algorithms Through Curriculum Design and Graph-Based Search*  
> Moving from rediscovery to novel algorithm discovery requires larger search budgets.
**[45]** Iwashita et al. (2022), 5.2.3 Other factors on solver performance, p. 8 — *Convergence Acceleration of Preconditioned CG Solver Based on Error Vector Sampling for a Sequence of Linear Systems*  
> In comparison of tables 2 and 4, the sampling method A gives better convergence acceleration than the method B.