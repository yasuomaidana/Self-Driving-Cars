# Outlier Rejection

## Image Feature

### Localization

Problem:

Find translation $\bm{T}= [t_u,t_v]$ in image coordinate system between image 1 and image 2.

* Matched feature pairs in images 1 and 2:
  * $f_i^{(1)},f_i^{(2)}|i \in [0,\dots,N]$
  * $f_i^{(1)} = (u_i^{(1)},v_i^{(1)})$
* Model:
  * $u_i^{(1)} + t_u = u_i^{(2)}$
  * $v_i^{(1)} + t_v = v_i^{(2)}$
* Solve using least squares:
  * $t_u=\frac{1}{N} \sqrt{\sum\limits_{i}\left(u_i^{(1)} - u_i^{(2)}\right)^2}$
  * $t_v=\frac{1}{N} \sqrt{\sum\limits_{i}\left(v_i^{(1)} - v_i^{(2)}\right)^2}$

## Random Sample Consensus (RANSAC)

Proposed by Fischler & Boiles 1981.

Algorithm

Initialization:

1. Given a Model, find the smallest number of samples, $\bm{M}$, from which the
model can be computed

Main Loop:

1. From your data, randomly select $\bm{M}$ samples
2. Compute model parameters using the selected $\bm{M}$ samples
3. Check how many samples from the rest of your data actually fits the model. We call this number the number of inliers $\bm{C}$
4. If $\bm{C}>$ **inlier ratio threshold or maximum iterations reached**, terminate
and return the best inlier set. Else, go back to step 2

Final Step:

1. Recompute model parameters from entire best inlier set

## Additional material

Forsyth, D.A. and J. Ponce (2003). Computer Vision: a modern approach (2nd edition). New Jersey: Pearson. Read section 19.1-19.3.
