# Attention-as-a-Laplace-Radon-Transform

Scaled dot-product attention is typically presented as a normalized exponential weighting
over pairwise query-key dot products. This paper compiles a transform-theoretic representation
of a single attention head by showing that the softmax normalizer and value-weighted numerator
are exact Laplace transforms of Radon projections of discrete measures on key space. In this
representation, each query determines a projection direction (a tomographic “tilt”) and a radial
Laplace parameter (a “depth” or effective temperature). The factorization yields a geometric
diagnostic toolkit for attention heads: (i) per-query tilt and radial statistics; (ii) three exact
null spaces acting as information bottlenecks (parameter-level invisibility, dataset limited-angle
tomography nulls, and forward-pass routing/value nulls); and (iii) an inverse pipeline for the
unnormalized objects via inverse Laplace followed by inverse Radon. Finally, by replacing
the Laplace kernel with other positive kernels, attention can be placed in a broader family of
generalized Radon integral transforms, enabling principled bias-variance tradeoffs and importing
tools from harmonic analysis.

Simply put:
This paper presents a discovery revealing that the standard attention mechanism is mathematically identical to the process of a tomographic scan (like a CT scan). 
It demonstrates that each query vector acts as a probe, defining a specific direction to project the key vectors onto (the Radon transform) and a sharpness for how to weight that projection (the Laplace transform). 
This powerful geometric viewpoint is not just a metaphor, providing an exact diagnostic toolkit and accurate predictions in each tested case.
Ultimately, this reframes attention as a specific instance of a broader family of integral transforms, opening a principled path to diagnosing existing models 
and designing novel attention mechanisms with controlled properties.
