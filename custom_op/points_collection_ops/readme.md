### points collection operation

Given sample points at each location in test image space,  we sample the deformable offset in DCN and accumulated the sample points and deformable offset.

Inputs: target offset $t$ , $t=R^{b\times N2 \times H \times W}$. deformable offset $d$ , $d=R^{b\times M2 \times h \times w}$.

outputs: accumulated offset $p$, $p=R^{b\times NM2 \times H \times W}$.

Top grad:$g= R^{b\times NM2 \times H \times W}$

The forward pass:

here we only focus on the last three dimensions, so the first dimension is eliminated.

For each $t_{ij}^{n}=(y_{ij}^n,x_{ij}^n),d_{pq}^m=(u_{pq}^m,v_{pq}^m)$, 

<font color=green> The ratio of height: $r_h=h/H$ </font>

<font color=green>The ratio of width: $r_w = w/W$</font>

~~The deformable offset:~~

~~$u_{qp}^m=u_{qp}^m/r_h$~~

~~$v_{qp}^m=v_{qp}^m/r_w$~~



for each value in output:
$$
p_{ij}^{2nm}=\sum_{p}^{h} \sum_{q}^{w} u_{pq}^m /r_h \max \left(0,1-\left|(i+y_{ij}^{n})*r_h -p \right|\right) \max \left(0,1-\left|(j+x_{ij}^n)*r_w -q \right|\right)+y_{ij}^n
$$

$$
p_{ij}^{2nm+1}=\sum_{p}^{h} \sum_{q}^{w} v_{pq}^m /r_w \max \left(0,1-\left|(i+y_{ij}^{n})*r_h -p \right|\right) \max \left(0,1-\left|(j+x_{ij}^n)*w_w -q \right|\right)+x_{ij}^n
$$



The backward pass:

for the input deformable offset:
$$
w=\sum_{p}^{h} \sum_{q}^{w}\max \left(0,1-\left|(i+y_{ij}^{n})*h/H -p \right|\right) \max \left(0,1-\left|(j+x_{ij}^n)*w/W -q \right|\right)
$$
$$
\frac{\partial p_{ij}^{2nm}}{\partial u_{pq}^{m}}=w/r_h * g_{ij}^{2nm}
$$

$$
\frac{\partial p_{ij}^{2nm+1}}{\partial v_{pq}^{m}}=w/r_w *g_{ij}^{2nm+1}
$$



for the target offset:
$$
\frac{\partial p_{ij}^{2nm}}{\partial y_{ij}^{n}}=1+\sum_{p}^{h} \sum_{q}^{w}u_{pq}^m \max \left(0,1-\left|(j+x_{ij}^n)*w/W -q \right|\right)
\left\{\begin{array}{ll} 
0 & \text { if }\left |(i+y_{ij}^{n})*h/H - p \right| \geq 1 \\
1 & \text { if } p \geq (i+y_{ij}^{n})*h/H \\
-1 & \text { if } p < (i+y_{ij}^{n})*h/H
\end{array}\right.
$$

$$
\frac{\partial p_{ij}^{2nm}}{\partial x_{ij}^{n}}=\sum_{p}^{h} \sum_{q}^{w}u_{pq}^m \max \left(0,1-
\left |(i+y_{ij}^{n})*h/H - p \right|\right)
\left\{\begin{array}{ll} 
0 & \text { if } \left|(j+x_{ij}^n)*w/W -q \right| \geq 1 \\
r_w/r_h & \text { if } q \geq (j+x_{ij}^n)*w/W \\
-r_w/r_h & \text { if } q < (j+x_{ij}^n)*w/W
\end{array}\right.
$$

and the other :
$$
\frac{\partial p_{ij}^{2nm+1}}{\partial y_{ij}^{n}}=\sum_{p}^{h} \sum_{q}^{w} v_{pq}^m \max \left(0,1-\left|(j+x_{ij}^n)*w/W -q \right|\right)
\left\{\begin{array}{ll} 
0 & \text { if }\left |(i+y_{ij}^{n})*h/H - p \right| \geq 1 \\
r_h/r_w & \text { if } p \geq (i+y_{ij}^{n})*h/H \\
-r_h/r_w & \text { if } p < (i+y_{ij}^{n})*h/H
\end{array}\right.
$$

$$
\frac{\partial p_{ij}^{2nm+1}}{\partial x_{ij}^{n}}=1+\sum_{p}^{h} \sum_{q}^{w} v_{pq}^m \max \left(0,1-
\left |(i+y_{ij}^{n})*h/H - p \right|\right)
\left\{\begin{array}{ll} 
0 & \text { if } \left|(j+x_{ij}^n)*w/W -q \right| \geq 1 \\
1 & \text { if } q \geq (j+x_{ij}^n)*w/W \\
-1 & \text { if } q < (j+x_{ij}^n)*w/W
\end{array}\right.
$$



### Test

OK!