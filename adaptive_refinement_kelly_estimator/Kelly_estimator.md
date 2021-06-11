Here the problem 

`- div(a(x) \nabla u) = f `

is solved over an L-shaped mesh with Dirichlet boundary conditions, using Kelly error estimator for adaptive mesh refinement. Successive refinements arise where the jump of `a(x)` occurs, i.e. along the circle of radius 0.5.
On the top of that, hanging nodes on quadrilaterals give rise to discontinuous basis functions along edges, so to get a conforming method, i.e. to get the solution to live on a true subspace of the original Sobolev space, we need to make sure every linear combination of basis functions (globally discontinuous) is still continuous. This is handled by the `AffineConstraints` class.
