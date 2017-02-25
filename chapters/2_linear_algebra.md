# 章2 线性代数

## 一、基本知识
1. 本书中所有的向量都是列向量的形式：
	$$
\mathbf{\vec x}=\begin{bmatrix}x_1\\\x_2\\\
\vdots \\\x_n\end{bmatrix}
	$$

2. 矩阵的`F`范数：设 \\(\mathbf A=(a\_{i,j})\_{m\times n}\\)
	$$||\mathbf A||_F=\sqrt{\sum\_{i,j}a\_{i,j}^{2}} $$
	它是向量的 \\(L_2\\) 范数的推广。

3. 矩阵的迹 \\(tr(\mathbf A)=\sum\_{i}a\_{i,i}\\)。其性质有：
	- \\(||\mathbf A||_F=\sqrt{tr(\mathbf A \mathbf A^{T})}\\)
	- \\(tr(\mathbf A)=tr(\mathbf A^{T})\\)
	- 假设 \\(\mathbf A\in \mathbb R^{m\times n},\mathbf B\in \mathbb R^{n\times m}\\)，则有：
		$$tr(\mathbf A\mathbf B)=tr(\mathbf B\mathbf A) $$
	- \\(tr(\mathbf A\mathbf B\mathbf C)=tr(\mathbf C\mathbf A\mathbf B)=tr(\mathbf B\mathbf C\mathbf A)\\)

## 二、向量操作

1. 一组向量 \\(\mathbf{\vec v}\_1,\mathbf{\vec v}\_2,\cdots,\mathbf{\vec v}\_n\\) 是线性相关的：指存在一组不全为零的实数 \\(a_1,a_2,\cdots,a_n\\)，使得： 
	$$\sum\_{i=1}^{n}a_i\mathbf{\vec v}\_i=\mathbf{\vec 0} $$

	一组向量 \\(\mathbf{\vec v}\_1,\mathbf{\vec v}\_2,\cdots,\mathbf{\vec v}\_n\\) 是线性无关的，当且仅当 \\(a_i=0,i=1,2,\cdots,n\\) 时，才有 
	$$\sum\_{i=1}^{n}a_i\mathbf{\vec v}\_i=\mathbf{\vec 0} $$

2. 一个向量空间所包含的最大线性无关向量的数目，称作该向量空间的维数。
3. 三维向量的点积：
	$$\mathbf{\vec u}\cdot\mathbf{\vec v} =u \_xv\_x+u\_yv\_y+u\_zv\_z = |\mathbf{\vec u}| | \mathbf{\vec v}| \cos(\mathbf{\vec u},\mathbf{\vec v})$$
	![dot](../imgs/2/dot.PNG)

4. 三维向量的叉积：
	$$ \mathbf{\vec w}=\mathbf{\vec u}\times \mathbf{\vec v}=\begin{bmatrix}\mathbf{\vec i}& \mathbf{\vec j}&\mathbf{\vec k}\\\
	u_x&u_y&u_z\\\
	v_x&v_y&v_z\\\
	\end{bmatrix}$$
	其中 \\(\mathbf{\vec i}, \mathbf{\vec j},\mathbf{\vec k}\\) 分别为 \\(x,y,z\\) 轴的单位向量。
	\\(\mathbf{\vec u}=u_x\mathbf{\vec i}+u_y\mathbf{\vec j}+u_z\mathbf{\vec k},\quad \mathbf{\vec v}=v_x\mathbf{\vec i}+v_y\mathbf{\vec j}+v_z\mathbf{\vec k}\\)
	- \\(\mathbf{\vec u} \\) 和 \\(\mathbf{\vec v}\\) 的叉积垂直于 \\(\mathbf{\vec u},\mathbf{\vec v}\\) 构成的平面，其方向符合右手规则。
	- 叉积的模等于 \\(\mathbf{\vec u},\mathbf{\vec v}\\) 构成的平行四边形的面积
	- \\(\mathbf{\vec u}\times \mathbf{\vec v}=-\mathbf{\vec v}\times \mathbf{\vec u}\\)
	- \\(\mathbf{\vec u}\times( \mathbf{\vec v} \times \mathbf{\vec w})=(\mathbf{\vec u}\cdot \mathbf{\vec w})\mathbf{\vec v}-(\mathbf{\vec u}\cdot \mathbf{\vec v})\mathbf{\vec w} \\)

	![cross](../imgs/2/cross.PNG)

5. 三维向量的混合积：
	$$[\mathbf{\vec u} \;\mathbf{\vec v} \;\mathbf{\vec w}]=(\mathbf{\vec u}\times \mathbf{\vec v})\cdot \mathbf{\vec w}= \mathbf{\vec u}\cdot (\mathbf{\vec v} \times \mathbf{\vec w})\\\
	=\begin{vmatrix}
	u_x&u_y&u_z\\\
	v_x&v_y&v_z\\\
	w_x&w_y&w_z
	\end{vmatrix}
	=\begin{vmatrix}
	u_x&v_x&w_x\\\
	u_y&v_y&w_y\\\
	u_z&v_z&w_z
	\end{vmatrix} $$
	- 其物理意义为：以 \\(\mathbf{\vec u} ,\mathbf{\vec v} ,\mathbf{\vec w}\\) 为三个棱边所围成的平行六面体的体积。 当 \\(\mathbf{\vec u} ,\mathbf{\vec v} ,\mathbf{\vec w}\\) 构成右手系时，该平行六面体的体积为正号。

6. 两个向量的并矢：给定两个向量 \\(\mathbf {\vec x}=(x_1,x_2,\cdots,x_n)^{T},   \mathbf {\vec y}= (y_1,y_2,\cdots,y_m)^{T}\\) ，则向量的并矢记作：
	$$\mathbf {\vec x}\mathbf {\vec y}  =\begin{bmatrix}
	x\_1y\_1&x\_1y\_2&\cdots&x\_1y\_m\\\
	x\_2y\_1&x\_2y\_2&\cdots&x\_2y\_m\\\
	\vdots&\vdots&\ddots&\vdots\\\
	x\_ny\_1&x\_ny\_2&\cdots&x\_ny\_m\\\
	\end{bmatrix}$$
	也记作  \\(\mathbf {\vec x}\otimes\mathbf {\vec y}\\) 或者 \\(\mathbf {\vec x} \mathbf {\vec y}^{T}\\)

## 三、矩阵运算
1. 给定两个矩阵 \\(\mathbf A=(a\_{i,j}) \in \mathbb R^{m\times n},\mathbf B=(b\_{i,j})  \in \mathbb R^{m\times n}\\) ，定义：
	- 阿达马积`Hadamard product`（又称作逐元素积）：
	$$\mathbf A \circ \mathbf B  =\begin{bmatrix}
	a\_{1,1}b\_{1,1}&a\_{1,2}b\_{1,2}&\cdots&a\_{1,n}b\_{1,n}\\\
	a\_{2,1}b\_{2,1}&a\_{2,2}b\_{2,2}&\cdots&a\_{2,n}b\_{2,n}\\\
	\vdots&\vdots&\ddots&\vdots\\\
	a\_{m,1}b\_{m,1}&a\_{m,2}b\_{m,2}&\cdots&a\_{m,n}b\_{m,n}
	\end{bmatrix}$$
	- 克罗内积`Kronnecker product`：
	$$\mathbf A \otimes \mathbf B =\begin{bmatrix}
	a\_{1,1}\mathbf B&a\_{1,2}\mathbf B&\cdots&a\_{1,n}\mathbf B\\\
	a\_{2,1}\mathbf B&a\_{2,2}\mathbf B&\cdots&a\_{2,n}\mathbf B\\\
	\vdots&\vdots&\ddots&\vdots\\\
	a\_{m,1}\mathbf B&a\_{m,2}\mathbf B&\cdots&a\_{m,n}\mathbf B
	\end{bmatrix}$$

2. 设 \\(\mathbf {\vec x},\mathbf {\vec a},\mathbf {\vec b},\mathbf {\vec c}\\) 为 \\(n\\) 阶向量， \\(\mathbf A,\mathbf B,\mathbf C,\mathbf X\\) 为 \\(n\\) 阶方阵，则：
  	$$
	\frac{\partial(\mathbf {\vec a}^{T}\mathbf {\vec x}) }{\partial \mathbf {\vec x} }=\frac{\partial(\mathbf {\vec x}^{T}\mathbf {\vec a}) }{\partial \mathbf {\vec x} }	=\mathbf {\vec a}$$
	$$\frac{\partial(\mathbf {\vec a}^{T}\mathbf X\mathbf {\vec b}) }{\partial \mathbf X }=\mathbf {\vec a}\mathbf {\vec b}^{T}=\mathbf {\vec a}\otimes\mathbf {\vec b}\in \mathbb R^{n\times n}$$
	$$\frac{\partial(\mathbf {\vec a}^{T}\mathbf X^{T}\mathbf {\vec b}) }{\partial \mathbf X }=\mathbf {\vec b}\mathbf {\vec a}^{T}=\mathbf {\vec b}\otimes\mathbf {\vec a}\in \mathbb R^{n\times n}$$
	$$\frac{\partial(\mathbf {\vec a}^{T}\mathbf X\mathbf {\vec a}) }{\partial \mathbf X }=\frac{\partial(\mathbf {\vec a}^{T}\mathbf X^{T}\mathbf {\vec a}) }{\partial \mathbf X }=\mathbf {\vec a}\otimes\mathbf {\vec a}$$
	$$\frac{\partial(\mathbf {\vec a}^{T}\mathbf X^{T}\mathbf X\mathbf {\vec b}) }{\partial \mathbf X }=\mathbf X(\mathbf {\vec a}\otimes\mathbf {\vec b}+\mathbf {\vec b}\otimes\mathbf {\vec a})$$
	$$
	\frac{\partial[(\mathbf A\mathbf {\vec x}+\mathbf {\vec a})^{T}\mathbf C(\mathbf B\mathbf {\vec x}+\mathbf {\vec b})]}{\partial \mathbf {\vec x}}=\mathbf A^{T}\mathbf C(\mathbf B\mathbf {\vec x}+\mathbf {\vec b})+\mathbf B^{T}\mathbf C(\mathbf A\mathbf {\vec x}+\mathbf {\vec a})
	$$
	$$
	\frac{\partial (\mathbf {\vec x}^{T}\mathbf A \mathbf {\vec x})}{\partial \mathbf {\vec x}}=(\mathbf A+\mathbf A^{T})\mathbf {\vec x}
	$$
	$$
	\frac{\partial[(\mathbf X\mathbf {\vec b}+\mathbf {\vec c})^{T}\mathbf A(\mathbf X\mathbf {\vec b}+\mathbf {\vec c})]}{\partial \mathbf X}=(\mathbf A+\mathbf A^{T})(\mathbf X\mathbf {\vec b}+\mathbf {\vec c})\mathbf {\vec b}^{T}
	$$
	$$
	\frac{\partial (\mathbf {\vec b}^{T}\mathbf X^{T}\mathbf A \mathbf X\mathbf {\vec c})}{\partial \mathbf X}=\mathbf A^{T}\mathbf X\mathbf {\vec b}\mathbf {\vec c}^{T}+\mathbf A\mathbf X\mathbf {\vec c}\mathbf {\vec b}^{T}
	$$
3. 如果 \\(f\\) 是一元函数，则：
	- 其逐元向量函数为：
	$$f(\mathbf{\vec x}) =(f(x_1),f(x_2),\cdots,f(x_n))^{T}$$
	- 其逐矩阵函数为：
	$$f(\mathbf X)=f(x\_{i,j}) $$

	其逐元导数分别为： 
	$$f^{\prime}(\mathbf{\vec x}) =(f^{\prime}(x_1),f^{\prime}(x_2),\cdots,f^{\prime}(x_n))^{T}\\\
	f^{\prime}(\mathbf X)=f^{\prime}(x\_{i,j})$$

4. 各种类型的偏导数：
	- 标量对标量的偏导数 $$\frac{\partial u}{\partial v}$$
	- 标量对向量（\\(n\\) 维向量）的偏导数 $$\frac{\partial u}{\partial \mathbf {\vec v}}=(\frac{\partial u}{\partial v\_1},\frac{\partial u}{\partial v\_2},\cdots,\frac{\partial u}{\partial v\_n})^{T}$$
	- 标量对矩阵(\\(m\times n\\) 阶矩阵)的偏导数
	$$
\frac{\partial u}{\partial \mathbf V}=\begin{bmatrix}
\frac{\partial u}{\partial V\_{1,1}}&\frac{\partial u}{\partial  V\_{1,2}}&\cdots&\frac{\partial u}{\partial  V\_{1,n}}\\\
\frac{\partial u}{\partial V\_{2,1}}&\frac{\partial u}{\partial  V\_{2,2}}&\cdots&\frac{\partial u}{\partial  V\_{2,n}}\\\
\vdots&\vdots&\ddots&\vdots\\\
\frac{\partial u}{\partial V\_{m,1}}&\frac{\partial u}{\partial  V\_{m,2}}&\cdots&\frac{\partial u}{\partial  V\_{m,n}}
\end{bmatrix}
	$$
	- 向量（\\(m\\) 维向量）对标量的偏导数 $$\frac{\partial \mathbf {\vec u}}{\partial v}=(\frac{\partial u\_1}{\partial v},\frac{\partial u\_2}{\partial v},\cdots,\frac{\partial u\_m}{\partial v})^{T}$$
	- 向量（\\(m\\) 维向量）对向量 (\\(n\\) 维向量) 的偏导数（雅可比矩阵，行优先）
	$$
\frac{\partial \mathbf {\vec u}}{\partial \mathbf {\vec v}}=\begin{bmatrix}
\frac{\partial u\_1}{\partial v\_1}&\frac{\partial u\_1}{\partial  v\_2}&\cdots&\frac{\partial u\_1}{\partial  v\_n}\\\
\frac{\partial u\_2}{\partial v\_1}&\frac{\partial u\_2}{\partial  v\_2}&\cdots&\frac{\partial u\_2}{\partial  v\_n}\\\
\vdots&\vdots&\ddots&\vdots\\\
\frac{\partial u\_m}{\partial v\_1}&\frac{\partial u\_m}{\partial  v\_2}&\cdots&\frac{\partial u\_m}{\partial  v\_n}
\end{bmatrix}
	$$
	> 如果为列优先，则为上面矩阵的转置
	- 矩阵(\\(m\times n\\) 阶矩阵)对标量的偏导数
	$$
	\frac{\partial \mathbf U}{\partial v}=\begin{bmatrix}
\frac{\partial U\_{1,1}}{\partial v}&\frac{\partial U\_{1,2}}{\partial  v}&\cdots&\frac{\partial U\_{1,n}}{\partial v}\\\
\frac{\partial U\_{2,1}}{\partial v}&\frac{\partial U\_{2,2}}{\partial  v}&\cdots&\frac{\partial U\_{2,n}}{\partial  v}\\\
\vdots&\vdots&\ddots&\vdots\\\
\frac{\partial U\_{m,1}}{\partial v}&\frac{\partial U\_{m,2}}{\partial v}&\cdots&\frac{\partial U\_{m,n}}{\partial  v}
\end{bmatrix}
	$$
	- 更复杂的情况依次类推。对于 \\(\frac{\partial \mathbf u}{\partial \mathbf v}\\)。根据`numpy`的术语：
		- 假设 \\(\mathbf u\\) 的 `ndim`（维度）为 \\(d_u\\) 
		>  对于标量， `ndim`为 0；对于向量， `ndim`为1；对于矩阵，`ndim`为 2 
		- 假设 \\(\mathbf v\\) 的 `ndim`为 \\(d_v\\)  
		- 则 \\(\frac{\partial \mathbf u}{\partial \mathbf v}\\) 的 `ndim`为 \\(d_u+d_v\\) 

5. 对于矩阵的迹，有下列偏导数成立：
	$$\frac{\partial [tr(f(\mathbf X))]}{\partial \mathbf X }=(f^{\prime}(\mathbf X))^{T}$$
	$$\frac{\partial [tr(\mathbf A\mathbf X\mathbf B)]}{\partial \mathbf X }=\mathbf A^{T}\mathbf B^{T} $$
	$$\frac{\partial [tr(\mathbf A\mathbf X^{T}\mathbf B)]}{\partial \mathbf X }=\mathbf B\mathbf A  $$
	$$\frac{\partial [tr(\mathbf A\otimes\mathbf X )]}{\partial \mathbf X }=tr(\mathbf A)\mathbf I$$
	$$ \frac{\partial [tr(\mathbf A\mathbf X \mathbf B\mathbf X)]}{\partial \mathbf X }=\mathbf A^{T}\mathbf X^{T}\mathbf B^{T}+\mathbf B^{T}\mathbf X \mathbf A^{T}  $$
	$$ \frac{\partial [tr(\mathbf  X^{T} \mathbf B\mathbf X \mathbf  C)]}{\partial \mathbf X }=(\mathbf B^{T}+\mathbf B)\mathbf X \mathbf C \mathbf C^{T}  $$
	$$ \frac{\partial [tr(\mathbf C^{T}\mathbf  X^{T} \mathbf B\mathbf X \mathbf  C)]}{\partial \mathbf X }=\mathbf B\mathbf X \mathbf C +\mathbf B^{T}\mathbf X \mathbf C^{T}  $$
	$$ \frac{\partial [tr(\mathbf A\mathbf  X \mathbf B\mathbf X^{T} \mathbf  C)]}{\partial \mathbf X }= \mathbf  A^{T}\mathbf  C^{T}\mathbf X\mathbf  B^{T}+\mathbf  C \mathbf  A \mathbf  X \mathbf  B$$
	$$ \frac{\partial [tr((\mathbf A\mathbf X\mathbf B+\mathbf C)(\mathbf A\mathbf X\mathbf B+\mathbf C))]}{\partial \mathbf X }= 2\mathbf A ^{T}(\mathbf A\mathbf X\mathbf B+\mathbf C)\mathbf B^{T}$$

6. 假设 \\(\mathbf U=\mathbf f(\mathbf X)\\) 是关于 \\(\mathbf X\\) 的矩阵值函数（\\(f:\mathbb R^{m\times n}\rightarrow \mathbb R^{m\times n}\\)），且 \\(g(\mathbf U)\\) 是关于 \\(\mathbf U\\) 的实值函数（\\(g:\mathbb R^{m\times n}\rightarrow \mathbb R \\)），则下面链式法则成立：
	$$\frac{\partial g(\mathbf U)}{\partial \mathbf X}=\left(\frac{\partial g(\mathbf U)}{\partial x\_{i,j}}\right)=\begin{bmatrix}
\frac{\partial g(\mathbf U)}{\partial x\_{1,1}}&\frac{\partial g(\mathbf U)}{\partial x\_{1,2}}\cdots&\frac{\partial g(\mathbf U)}{\partial x\_{1,n}}\\\
\frac{\partial g(\mathbf U)}{\partial x\_{2,1}}&\frac{\partial g(\mathbf U)}{\partial x\_{2,2}}\cdots&\frac{\partial g(\mathbf U)}{\partial x\_{2,n}}\\\
\vdots&\vdots&\ddots&\vdots\\\
\frac{\partial g(\mathbf U)}{\partial x\_{m,1}}&\frac{\partial g(\mathbf U)}{\partial x\_{m,2}}\cdots&\frac{\partial g(\mathbf U)}{\partial x\_{m,n}}\\\
\end{bmatrix}\\\
=\left(\sum\_{k}\sum\_{l}\frac{\partial g(\mathbf U)}{\partial u\_{k,l}}\frac{\partial u\_{k,l}}{\partial x\_{i,j}}\right)\\\
	=tr\left[\left(\frac{\partial g(\mathbf U)}{\partial \mathbf U}\right)^{T}\frac{\partial \mathbf U}{\partial x\_{i,j}}\right]
	 $$