# 章2 线性代数
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