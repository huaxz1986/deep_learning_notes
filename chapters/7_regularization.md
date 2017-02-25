# 章7 正则化

##一、 基本概念
1. 目前有多种正则化策略：
	- 有些策略是向机器学习模型添加额外的约束，如增加对参数的限制
	- 有些策略是向目标函数增加额外项，此时对应于参数值的软约束

	有时候这些策略代表了特定类型的先验知识。有时候这些策略代表了对模型的偏好（比如偏好简单的模型）

2. 在深度学习中，大多数正则化策略都是基于对估计量进行正则化：以偏差的增加来换取方差的减少
	- 一个有效的正则化能显著降低方差，而不会过度增加偏差

3. 我们几乎从来无法知晓真实数据的生成过程，因此我们永远不知道被估计的模型族是否包含真实的生成过程。
	- 在实际的深度学习场景中，最好的拟合模型（基于最小泛化误差）是一个适当正则化的大型模型

##二、 参数范数正则化
1. 一些正则化方法通过对目标函数 \\(J\\) 添加一个参数范数正则化项 \\(\Omega(\vec\theta)\\) 来限制模型的容量`capacity` 。正则化之后的目标函数为 \\(\tilde J\\)：
	$$ \tilde J(\vec\theta;\mathbf X,\mathbf{\vec y})= J(\vec\theta;\mathbf X,\mathbf{\vec y})+\alpha \Omega(\vec\theta)$$ 
	- \\(\alpha \in [0,\infty)\\) 为正则化项的系数。它衡量正则化项 \\(\Omega(\vec\theta)\\) 和标准目标函数 \\(J\\) 的比重。
		- \\(\alpha=0\\) 则没有正则化
		- \\(\alpha\\) 越大，则正则化项越重要
	- 如果最小化 \\(\tilde J\\)，则会同时降低 \\(J\\) 和参数 \\(\vec\theta\\) 的规模

2. 选择不同的 \\(\Omega\\)   的形式会产生不同的解。常见的有 \\(L\_2\\) 正则化和 \\(L\_1\\)  正则化

### 1. L2 正则化

1.  \\(L\_2\\) 正则化通常被称作岭回归或者`Tikhonov`正则化。
	- 正则化项为 \\(\Omega(\vec\theta)=\frac 12|| \vec\theta||\_2^{2}\\)。系数 \\(\frac 12\\) 是为了使得导数的系数为 1.
	- 该正则化形式倾向于使得参数 \\(\vec\theta\\) 更接近零

2. 假设 \\(\vec\theta\\) 参数就是权重 \\(\mathbf{\vec w}\\)，没有偏置参数。则：
	$$\tilde J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})=J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})+\frac {\alpha}{2}\mathbf{\vec w}^{T}\mathbf{\vec w}$$
	对应的梯度为
	$$\nabla\_{\mathbf{\vec w}}\tilde J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})=\nabla\_{\mathbf{\vec w}} J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})+\alpha \mathbf{\vec w}$$

3. 使用梯度下降法来更新权重，给出权重的更新公式为：
	$$ \mathbf{\vec w}\leftarrow \mathbf{\vec w}-\epsilon(\nabla\_{\mathbf{\vec w}} J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})+\alpha \mathbf{\vec w})$$
	即 \\( \mathbf{\vec w}\leftarrow (1-\epsilon\alpha) \mathbf{\vec w}-\epsilon \nabla\_{\mathbf{\vec w}} J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y}) \\)
	
	可以看到： 经过 \\(L\_2\\) 正则化之后，采用梯度下降法时：每一步执行梯度更新之前，会对权重向量乘以一个常数因子来收缩权重向量。这就是 \\(L\_2\\) 正则化对于单步更新的影响。

4. 令 \\(\mathbf{\vec w}^{\*}=\arg\min\_{\mathbf{\vec w}}J(\mathbf{\vec w})\\)，它就是无正则化项时使得目标函数最小的权重向量。在  \\(\mathbf{\vec w}^{\*}\\) 的邻域内泰勒展开（根据极小值的条件，有梯度为零）：
		$$\hat J(\mathbf{\vec w})=J(\mathbf{\vec w}^{\*}) +\frac 12(\mathbf{\vec w}-\mathbf{\vec w}^{\*})^{T}\mathbf H(\mathbf{\vec w}-\mathbf{\vec w}^{\*})$$
		\\(\mathbf H\\) 为 \\(J\\) 在  \\(\mathbf{\vec w}^{\*}\\) 处的海森矩阵。  \\(\hat J(\mathbf{\vec w})\\) 的梯度为：\\(\nabla\_{\mathbf{\vec w}} \hat J(\mathbf{\vec w})=\mathbf H(\mathbf{\vec w}-\mathbf{\vec w}^{\*})\\) 

5. 令 \\(\tilde{\mathbf{\vec w}}^{\*}=\arg\min\_{\mathbf{\vec w}}\tilde J(\mathbf{\vec w})\\)，它就是有正则化项时使得目标函数最小的权重向量。此时梯度为零：
	$$\nabla\_{\mathbf{\vec w}}\tilde J(\tilde{\mathbf{\vec w}}^{\*})=\nabla\_{\mathbf{\vec w}} J(\mathbf{\tilde{\mathbf{\vec w}}^{\*}})+\alpha\tilde{\mathbf{\vec w}}^{\*}=\mathbf{\vec 0} $$
	将 \\(\hat J\\) 替换 \\(J\\) 有：
	$$\mathbf H(\tilde{\mathbf{\vec w}}^{\*}-\mathbf{\vec w}^{\*})+\alpha\tilde{\mathbf{\vec w}}^{\*}=\mathbf{\vec 0} $$
	$$\rightarrow (\mathbf H+\alpha\mathbf I)\tilde{\mathbf{\vec w}}^{\*}=\mathbf H \mathbf{\vec w}^{\*}$$
	$$ \rightarrow \tilde{\mathbf{\vec w}}^{\*}=(\mathbf H+\alpha\mathbf I)^{-1}\mathbf H \mathbf{\vec w}^{\*}$$
	- 当 \\(\alpha \rightarrow 0\\) 时， \\(\tilde{\mathbf{\vec w}}^{\*} \rightarrow  \mathbf{\vec w}^{\*}\\)
	- 因为 \\(\mathbf H\\) 是实对称矩阵，对其进行特征值分解。特征值组成对角矩阵 \\(\mathbf  \Lambda\\)，对应的特征向量组成正交矩阵  \\(\mathbf Q\\)。
		$$\mathbf  \Lambda=\begin{bmatrix}\lambda_1&0&\cdots&0\\\
			0&\lambda_2&\cdots&0\\\
		\vdots&\vdots&\ddots&\vdots\\\
		0&0&\cdots&\lambda_n 
		\end{bmatrix} $$
  		\\(\mathbf H=\mathbf Q\mathbf  \Lambda\mathbf Q^{T}\\)。于是有：
		$$\tilde{\mathbf{\vec w}}^{\*}=(\mathbf H+\alpha\mathbf I)^{-1}\mathbf H \mathbf{\vec w}^{\*} \\\
		=(\mathbf Q\mathbf  \Lambda\mathbf Q^{T}+\alpha\mathbf I)^{-1}\mathbf Q\mathbf  \Lambda\mathbf Q^{T} \mathbf{\vec w}^{\*} \\\
		=[\mathbf Q(\mathbf  \Lambda+ \alpha\mathbf I)\mathbf Q^{T}]^{-1}\mathbf Q\mathbf  \Lambda\mathbf Q^{T} \mathbf{\vec w}^{\*} \\\
		= \mathbf Q(\mathbf  \Lambda+ \alpha\mathbf I)^{-1}\mathbf  \Lambda\mathbf Q^{T} \mathbf{\vec w}^{\*} 
		$$
		其中：
		$$(\mathbf  \Lambda+ \alpha\mathbf I)^{-1}\mathbf  \Lambda=\begin{bmatrix}
		\frac{\lambda_1}{\lambda_1+\alpha}&0&\cdots&0\\\
		0&\frac{\lambda_2}{\lambda_2+\alpha}&\cdots&0\\\
		\vdots&\vdots&\ddots&\vdots\\\
		0&0&\cdots&\frac{\lambda_n}{\lambda_n+\alpha}
		\end{bmatrix}  $$
		
	可以看到 \\(L\_2\\) 的整体效果是：沿着 \\(\mathbf H\\) 的特征向量所定义的轴来缩放  \\(\mathbf{\vec w}^{\*} \\)。
	-  \\(\mathbf H\\) 的第 \\(i\\) 个特征向量对其的  \\(\mathbf{\vec w}^{\*} \\) 根据 \\(\frac{\lambda_i}{\lambda_i+\alpha}\\) 因子缩放。
	- 沿着 \\(\mathbf H\\) 特征值较大的方向受到正则化的影响较小
	- 当 \\(\lambda_i \ll \alpha \\) 的方向对应的权重分量将被缩小到几乎为零。

	如下所示：实线椭圆表示  \\(J\\) 的等值线，虚线圆表示正则化项  \\(\frac {\alpha}{2}\mathbf{\vec w}^{T}\mathbf{\vec w}\\)  的等值线。
	- 在 \\(\tilde{\mathbf{\vec w}}^{\*}\\) 点（也就是图中的 \\(\tilde w\\) 点），二者达到平衡（使得二者之和最小）
	- \\(J\\) 的海森矩阵第一维的特征值很小（沿着 \\(w_1\\) 方向的曲率半径较小；曲率半径越小，曲率越大，特征值越小）。所以当从   \\( \mathbf{\vec w} ^{\*}\\) 点水平移动时， \\(J\\) 不会增加太多。因为 \\(J\\) 对这个方向没有强烈的偏好，所以正则化项对于该轴具有强烈的影响：正则化项将 \\(w_1\\) 拉向零
	- 第二维的特征值较大。 \\(J\\)  对于 \\(w_2\\) 的变化非常敏感。因此正则化项对于该轴影响较小。
	>- 因为沿着水平方向，一个较大的偏移只会对 \\(J\\) 产生一个较小的变化。因此正则化项倾向于从  \\( \mathbf{\vec w} ^{\*}\\) 点水平移动
	>- 曲率半径是曲率的导数。对于椭圆 \\(\frac{w_1^{2}}{a^{2}}+\frac{w_2^{2}}{b^{2}}=1\\)，上下顶点的曲率半径为 \\(\frac{a^{2}}{b}\\)，左右顶点的曲率半径为 \\(\frac{b^{2}}{a}\\)。海森矩阵的特征值为 \\(\lambda_1= \frac{2}{a^{2}},\lambda_2= \frac{2}{b^{2}}\\)

	![L2](../imgs/7/L2.PNG)

6. \\(L_2\\) 正则化告诉我们：
	- 只有显著减小目标函数 \\(J\\) 的那个方向的参数会相对保留下来
	- 无助于减小目标函数 \\(J\\) 的方向（该方向上特征值较小），因为在这个方向上移动不会显著改变梯度，因此这个不重要方向上的分量会因为正则化的引入而被衰减掉。

7.  考虑 \\(L\_2\\) 正则化的一个特例：线性回归的 \\(L\_2\\) 正则化。采用平方误差作为代价函数：
	$$ J=(\mathbf X\mathbf{\vec w}-\mathbf{\vec y})^{T}(\mathbf X\mathbf{\vec w}-\mathbf{\vec y})$$
	 $$\tilde J =J +\frac {\alpha}{2}\mathbf{\vec w}^{T}\mathbf{\vec w} 
	=(\mathbf X\mathbf{\vec w}-\mathbf{\vec y})^{T}(\mathbf X\mathbf{\vec w}-\mathbf{\vec y})+\frac {\alpha}{2}\mathbf{\vec w}^{T}\mathbf{\vec w}$$
	- \\(\mathbf{\vec w}^{\*}=\arg\min\_{\mathbf{\vec w}}J(\mathbf{\vec w})\\)，解析解为：
	$$\mathbf{\vec w}^{\*}= (\mathbf X^{T}\mathbf X)^{-1}\mathbf X^{T}\mathbf{\vec y}$$
	-  \\(\tilde{\mathbf{\vec w}}^{\*}=\arg\min\_{\mathbf{\vec w}}\tilde J(\mathbf{\vec w})\\)，解析解为：
	$$\tilde{\mathbf{\vec w}}^{\*}=(\mathbf X^{T}\mathbf X+\alpha\mathbf I)^{-1}\mathbf X^{T}\mathbf{\vec y} $$

	样本的协方差矩阵为 \\(\mathbf \Sigma=\frac 1m\mathbf X^{T}\mathbf X\\)， \\(m\\) 为样本数量。因此可以看到：\\(\mathbf X^{T}\mathbf X\\) 的对角线对应于每个输入特征的方差。而 \\(\mathbf X^{T}\mathbf X+\alpha\mathbf I\\) 在对角线上增加了 \\(\alpha\\)。 \\(L_2\\) 正则化使得方差较小的特征对应的权重被收缩。
	> 方差远大于 \\(\alpha\\)  的特征受\\(L_2\\) 正则化影响较小。只有方差接近甚至小于  \\(\alpha\\)  的特征受影响较大。

### 2. L1 正则化
1. 模型参数 \\(\mathbf{\vec w}\\) 的 \\(L_1\\) 的正则化形式为：
	$$\Omega(\vec\theta)=||\mathbf{\vec w}||\_1=\sum\_{i}|w_i| $$
	即各个参数的绝对值之和。
	 
2. \\(L_1\\)  正则化后的目标函数 \\(\tilde J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})\\)：
	$$\tilde J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})=J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})+ \alpha ||\mathbf{\vec w}||\_1$$
	对应的梯度为
	$$\nabla\_{\mathbf{\vec w}}\tilde J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})=\nabla\_{\mathbf{\vec w}} J(\mathbf{\vec w};\mathbf X,\mathbf{\vec y})+\alpha \text{sign}(\mathbf{\vec w})$$
	其中 \\(\text{sign}(\cdot)\\) 函数取自变量的符号：如果自变量大于零，则取值为 1；如果自变量小于零，则取值为 -1；如果自变量为零，则取值为零。
	- \\(L_1\\) 正则化项对于梯度的贡献不再是线性地缩放每个 \\(w_i\\) （\\(L_2\\) 正则化项的效果），而是与 \\(\text{sign}(w_i)\\) 同号的常数因子。

3. 令 \\(\mathbf{\vec w}^{\*}=\arg\min\_{\mathbf{\vec w}}J(\mathbf{\vec w})\\)，它就是无正则化项时使得目标函数最小的权重向量。和 \\(L_2\\) 正则化中的推导相同，有：
		$$\nabla\_{\mathbf{\vec w}} \hat J(\mathbf{\vec w})=\mathbf H(\mathbf{\vec w}-\mathbf{\vec w}^{\*})$$
		\\(\mathbf H\\) 为 \\(J\\) 在  \\(\mathbf{\vec w}^{\*}\\) 处的海森矩阵。 其中 \\(\mathbf{\vec w}\\)  在  \\(\mathbf{\vec w}^{\*}\\) 的附近。 

	由于 \\(L_1\\) 正则化项在一般的海森矩阵情况下无法得到直接的代数表达式。因此我们进一步假设海森矩阵是对角矩阵，即：
	$$
\mathbf H=\begin{bmatrix}
	H\_{1,1}&0&\cdots&0\\\
	0&H\_{2,2}&\cdots&0\\\
	\vdots&\vdots&\ddots&\vdots\\\
	0&0&\cdots&H\_{n,n}\\\
	\end{bmatrix}
	$$
	其中 \\(H\_{i,i}\gt 0,i=1,2,\cdots,n\\)
	> 如果用于线性回归问题的数据已经被预处理（如使用`PCA`），去除了输入特征之间的相关性，则这一假设成立。

	于是：
	$$\hat J(\mathbf{\vec w})=J(\mathbf{\vec w}^{\*}) +\frac 12(\mathbf{\vec w}-\mathbf{\vec w}^{\*})^{T}\mathbf H(\mathbf{\vec w}-\mathbf{\vec w}^{\*})\\\
	\tilde J(\mathbf{\vec w})=\hat J(\mathbf{\vec w})+\alpha ||\mathbf{\vec w}||\_1=J(\mathbf{\vec w}^{\*})+\sum\_i\left[\frac 12 H\_{i,i}(w_i-w\_i^{\*})^{2}+\alpha|w_i|\right]$$
	对于 \\(\mathbf{\vec w}\\) 来讲 ， \\(J(\mathbf{\vec w}^{\*})\\) 为常量。因此 \\(\tilde J(\mathbf{\vec w})\\) 的最小值由 \\(\sum\_i\left[\frac 12 H\_{i,i}(w_i-w\_i^{\*})^{2}+\alpha|w_i|\right]\\) 决定。考虑每一个维度 \\(i\\)，可以考虑最优化目标：
	$$\tilde w\_i^{\*}= \arg\min\_{w\_i}\left[\frac 12 H\_{i,i}(w_i-w\_i^{\*})^{2}+\alpha|w_i|\right]$$
	得到解析解：
	$$\tilde w\_i^{\*}=\text{sign}(w\_i^{\*}) \max\left\\{|w\_i^{\*}|-\frac{\alpha}{H\_{i,i}},0\right\\}$$
	- 考虑 \\(w\_i^{\*} \gt 0\\) 的情况。此时有两种可能：
		- \\(w\_i^{\*} \le \frac{\alpha}{H\_{i,i}}\\)：则  \\(\tilde w\_i^{\*}=0\\)。表示 \\(L_1\\) 正则化项将 \\(w_i\\) 推向 0
		- \\(w\_i^{\*} \gt \frac{\alpha}{H\_{i,i}}\\)：则  \\(\tilde w\_i^{\*}= w\_i^{\*} -\frac{\alpha}{H\_{i,i}}\\)。此时 \\(L_1\\) 正则化项并不会将 \\(w_i\\) 推向 0，而是向零的方向推动了 \\(\frac{\alpha}{H\_{i,i}}\\) 的距离。

	- 考虑 \\(w\_i^{\*} \lt 0\\) 的情况。此时有两种可能：
		- \\( w\_i^{\*}  \ge  -\frac{\alpha}{H\_{i,i}}\\)：则  \\(\tilde w\_i^{\*}=0\\)。表示 \\(L_1\\) 正则化项将 \\(w_i\\) 推向 0
		- \\(w\_i^{\*} \lt -\frac{\alpha}{H\_{i,i}}\\)：则  \\(\tilde w\_i^{\*}= w\_i^{\*} +\frac{\alpha}{H\_{i,i}}\\)。此时 \\(L_1\\) 正则化项并不会将 \\(w_i\\) 推向 0，而是向零的方向推动了 \\(\frac{\alpha}{H\_{i,i}}\\) 的距离。

	> 如果使用 \\(L_2\\) 正则化，则解为 
	>$$\tilde w\_i^{\*}=\frac{H\_{i,i}}{H\_{i,i}+\alpha}w\_i^{\*} $$

4. \\(L_1\\) 正则化项更容易产生稀疏(`sparse`)解。
	- \\(L_1\\) 正则化的稀疏性与 \\(L_2\\) 不同。 \\(L_2\\)  正则化并不会导致参数的解变得稀疏。
	- \\(L_1\\) 正则化的这一特征已经被广泛地用作特征选择： \\(L_1\\) 正则化使得部分特征子集的权重为零，表明相应的特征可以被安全地忽略
	> 著名的`LASSO`模型将 \\(L_1\\) 正则化和线性模型结合，使用最小平方误差作为代价函数。

5. 许多正则化策略可以被解释为最大后验估计`MAP`贝叶斯推断：
	$$\vec\theta\_{MAP} =\arg\max\_{\vec\theta}p(\vec\theta\mid \mathbf{\vec x})=\arg\max\_{\vec\theta}p(\mathbf{\vec x} \mid \vec\theta )+\log p(\vec\theta)$$
	>最大化后验估计等价于最小化代价函数

	-  \\(L_2\\) 正则化项：参数的先验分布为高斯分布：
		$$\log p(\mathbf{\vec w})=\log \mathcal N(\mathbf{\vec w};0,\frac 1\alpha \mathbf I) =-\frac {\alpha}{2}\mathbf{\vec w}^{T}\mathbf{\vec w}+\frac n2 \log\frac{\alpha}{2\pi}$$
		忽略  \\(\frac n2 \log\frac{\alpha}{2\pi}\\) 项，因为它们与 \\(\mathbf{\vec w}\\) 无关。
	-  \\(L_1\\) 正则化项：参数的先验分布为各向同性拉普拉斯分布 ：
		$$\log p(\mathbf{\vec w})=\sum\_{i}\log \text{Laplace}(w_i;0,\frac 1\alpha) =-\alpha||\mathbf{\vec w}||_1+n\log\frac{\alpha}{2}$$
		忽略  \\(n\log\frac{\alpha}{2}\\) 项，因为它们与 \\(\mathbf{\vec w}\\) 无关。
	> 一维拉普拉斯分布的概率密度函数为 $$ f(x;\mu,b)=\frac{1}{2b}\exp\left(-\frac{|x-\mu|}{b}\right)$$
	- 更复杂的正则化项可以通过先验分布为混合高斯分布得到 

## 三、 约束最优化正则化 
1. 我们可以通过添加一个约束来实现正则化：
	$$
	\min\_{\vec\theta} J(\vec\theta;\mathbf X,\mathbf{\vec y})\\\
	st.\quad \Omega(\vec\theta) \lt k
	$$
	其中 \\(k\\) 为一个常数。 我们构建广义拉格朗日函数： 
	$$\mathcal L(\vec\theta,\alpha)=J(\vec\theta)+\alpha(\Omega(\vec\theta) - k)$$ 
	则上述约束最优化问题的解由下式给出：
	$$ \vec\theta^{\*}=\arg\min\_{\vec\theta}\max\_{\alpha,\alpha\gt 0}\mathcal L(\vec\theta,\alpha)$$
	- 假设 \\(\alpha\\) 的解为 \\(\alpha^{\*}\\)，固定 \\(\alpha^{\*}\\) 则：
		$$\vec\theta^{\*}=\arg\min\_{\vec\theta}J(\vec\theta)+\alpha^{\*}\Omega(\vec\theta) $$
		这和最小化 \\(\tilde J\\) 的正则化是相同的。因此我们可以将参数范数正则化视为对参数强加的约束
		- 如果 \\(\Omega\\) 是 \\(L\_2\\) 范数，则权重就是被约束在一个 \\(L\_2\\) 球中
		- 如果 \\(\Omega\\) 是 \\(L\_1\\) 范数，则权重就是被约束在一个 \\(L\_1\\) 限制的区间中

2. 有时候我们可以使用显式的约束（而不是使用正则化项）。此时可以修改梯度下降算法：
	- 首先计算 \\(J(\vec\theta)\\) 的下降步
	- 然后将 \\(\vec\theta\\) 投影到满足 \\(\Omega(\vec\theta) \lt k\\) 的最近点

3. 使用重投影的显式的限制而不是使用正则化项有两个好处：
	-  采用正则化项后，当 \\(\vec\theta\\) 较小时，容易使得非凸优化的过程陷入局部极小值。考虑当训练神经网络时，训练带有几个“死单元”的神经网络。这些死单元不会对网络学到的函数有太大的贡献。
		- 当使用权重范数的正则化时，较小的权重可能是局部最优的
		- 当使用显式的限制时，算法不鼓励权重接近原点，因此工作的较好

	- 使用显式的限制对优化过程增加了一定的稳定性。当使用了较高的学习率时，很可能进入了正反馈：
		- 较大的权重产生了较大的梯度
		- 较大的梯度诱发权重的更大的更新

		如果这些更新持续增加了权重的大小，则 \\(\vec\theta\\) 就会迅速增大，直到溢出。显式的限制可以防止这种反馈环引起的权重的无限制持续增加。
	 
4. `Srebro and Shraibman`对于神经网络引入的正则化策略为：约束神经网络的权重矩阵每列的范数，而不是限制整个权重矩阵的`Frobenius`范数。
	- 分别限制每一列的范数可以防止某一个隐单元有非常大的权重
	- 在实践中，列范数的限制总是通过重投影的显式约束来实现

## 四、 正则化和欠定问题
1. 机器学习的许多线性问题（包括线性回归和`PCA`），都依赖于求 \\(\mathbf X^{T}\mathbf X\\) 的逆矩阵。
	- 当 \\(\mathbf X^{T}\mathbf X\\) 是可逆时，该问题有解析解。
	- 当 \\(\mathbf X^{T}\mathbf X\\) 是奇异矩阵时，该问题是欠定的。此时，可以考虑正则化形式：求解 \\(\mathbf X^{T}\mathbf X+\alpha\mathbf I\\) 的逆矩阵

2. 大多数形式的正则化能够用于欠定问题。使用正则化来解决欠定问题的思想超出了机器学习的范畴。
	- 如`Moore-Penrose`求解欠定线性方程， \\(\mathbf X\\) 伪逆的一个定义：
		$$\mathbf X^{+}=\lim\_{\alpha \rightarrow 0}(\mathbf X^{T}\mathbf X+\alpha \mathbf I)^{-1} \mathbf X^{T}$$

## 五、 数据集增强
1. 提高机器学习模型泛化能力的一个最佳实践是：采用更多的数据来训练。但是现实中，我们拥有的数据量有限。
	
	- 解决这个问题的一种方案是：创建一些虚假的数据，并把它添加到训练集

2. 对于分类问题来说，创建虚假的数据是非常简单的。

	对于一个分类器，它将高维的输入 \\(\mathbf{\vec x}\\) 映射到类别 \\(y\\)。这意味着：
	- 这种映射规则是不随坐标系的改变而改变的
	- 因此我们可以通过线性变换，将训练集中的 \\((\mathbf{\vec x},y)\\) 变换为  \\((\mathbf{\vec x}^{\prime},y)\\) ，从而产生了新的数据  \\((\mathbf{\vec x}^{\prime},y)\\) 

3. 对于某些任务来说，创建虚假数据非常困难。如在密度估计任务中：除非你知道了密度函数，否则无法产生新的虚假数据。
4. 对图像识别这种分类问题来说，数据集增强特别有效。
	- 将训练图像沿着每个方向平移几个像素产生新的图像。这种操作通常可以大大改善泛化性能
	- 对训练图像进行旋转或者缩放也被证明非常有效。

	注意：
	- 某些线性变换会改变正确的类别。如字符识别任务中， `b/d`以及`6/9`的图像， 水平翻转和旋转 180度并不是一个合适的数据集增强方式。
	- 某些线性变换难以执行。如平面外的绕轴转动难以通过简单的几何运算在输入像素上实现。

5. 数据集增强在语音识别任务中也是有效的

6. 在神经网络的输入层注入噪声也可以视作数据增强的一种形式。
	- 对许多分类甚至一些回归任务，在输入上添加小的随机噪声之后，任务也能够顺利解决。
	- 一个训练好的神经网络被证明是对噪声不是非常健壮。改善神经网络健壮性的方法之一是：简单地将随机噪声施加到输入上，再进行训练。
		- 输入噪声注入是一些无监督学习算法的一部分，如降噪自动编码器
		- 噪声被添加到隐单元也是可行的。这被视为在多个抽象层上进行的数据集增强
		- `Poole et al.(2014)`表明：当仔细调整噪声的幅度之后，该方法非常高效
		- 本章后面的`dropout`正则化策略可以被看作是通过乘上噪声来构建新的输入的过程

7. 当比较机器学习算法基准测试的结果时，是否采用了数据集增强必须考虑在内。
	- 通常情况下，人工设计的数据集增强方案可以大大减少机器学习技术的泛化误差
	- 当两个机器学习算法的泛化性能比较时，应该确保这两个算法使用同一个人工设计的数据集增强方案进行评估
	
8. 要注意数据集增强和预处理的区别：如随机裁剪图像的操作被认为是独立的预处理步骤，而不是数据集增强。

## 六、 噪声鲁棒性
1. 有两种添加噪声的策略：
	- 一种策略是：将噪声作用于输入的数据集（这是前文介绍的一种数据集增强方法）
		- 对于某些模型，在输入上注入方差极小的噪音：等价于对权重施加范数正则化（`Bishop,1995a,b`）
		- 在一般情况下，噪声注入远比简单地收缩参数强大，尤其是噪声被添加到隐单元上时。
	- 另一种策略是：将噪音作用于权重。这也是一种正则化策略。这项技术主要用于循环神经网络。
		- 这种做法可以解释为一个关于权重的贝叶斯推断的随机实现。贝叶斯处理的训练过程将权重视作不确定的随机变量（拥有某个概率分布）。向权重注入噪声是对该随机变量采样得到的一个随机值。

2. 在某些假设下，将噪声注入权重等价于传统的正则化形式。假设我们有一个 \\(l\\) 层的标准的深度前馈神经网络，我们将噪声注入到该网络的权重。我们假设  \\( p\_{model}(y\mid \mathbf{\vec x};\vec\theta)=\mathcal N(y;f(\mathbf{\vec x};\vec\theta),\mathbf I)\\) ，则有：

	$$J(\vec\theta)=-\mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}} \log p\_{model}(y\mid \mathbf{\vec x};\vec\theta)\\\
	=\frac 12 \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}||y-f(\mathbf{\vec x};\vec\theta)||^{2}+\text{const} 
	  $$

	常数项包含了高斯分布的方差（与 \\(\vec\theta\\) 无关）。 于是重写为：
	$$J(\vec\theta)=\frac 12 \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}||y-f(\mathbf{\vec x};\vec\theta)||^{2} $$
	
	假设每个权重添加一个随机扰动 \\(\epsilon\_{\mathbf W}\sim \mathcal N(\epsilon;\mathbf{\vec 0},\eta\mathbf I)\\)。假设添加扰动之后的模型为 \\( \tilde p\_{model}(y\mid \mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})=\mathcal N(y;\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}),\mathbf I)\\)。我们假设有  \\(\mathbb E\_{p(\epsilon\_{\mathbf W})}  \tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})=f(\mathbf{\vec x};\vec\theta)\\)

	于是：
	$$\tilde J(\vec\theta)=\frac 12 \mathbb E\_{p(\mathbf{\vec x},y,\epsilon\_{\mathbf W})}||y-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})||^{2} \\\
	=\mathbb E\_{p(\epsilon\_{\mathbf W})}\left[\frac 12 \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}||y-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})||^{2}\right]\\\
	= \frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}||y-f(\mathbf{\vec x};\vec\theta)+f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})||^{2}\\\
	=\frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}\left[(y-f(\mathbf{\vec x};\vec\theta))^{2}+(f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))^{2}+\\\2(y-f(\mathbf{\vec x};\vec\theta))(f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))\right]\\\
	=\frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}} (y-f(\mathbf{\vec x};\vec\theta))^{2}+\frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}(f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))^{2}\\\
	+\mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}(y-f(\mathbf{\vec x};\vec\theta))\mathbb E\_{p(\epsilon\_{\mathbf W})} (f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))
	$$
	根据：
	 $$\frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}} (y-f(\mathbf{\vec x};\vec\theta))^{2}=\frac 12 \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}} (y-f(\mathbf{\vec x};\vec\theta))^{2} =J(\vec\theta)$$
	$$ \mathbb E\_{p(\epsilon\_{\mathbf W})} (f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))=f(\mathbf{\vec x};\vec\theta)-\mathbb E\_{p(\epsilon\_{\mathbf W})}\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})=0 $$
	于是有：
	$$\tilde J(\vec\theta)= J(\vec\theta)+\frac 12\mathbb E\_{p(\epsilon\_{\mathbf W})} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}(f(\mathbf{\vec x};\vec\theta)-\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W}))^{2}+0$$

	我们将  \\(\tilde f\\) 在 \\(\mathbf W\\) 处泰勒展开，有：
	$$\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})=f(\mathbf{\vec x};\vec\theta)+\nabla\_{\mathbf W}f(\mathbf{\vec x};\vec\theta)^{T} \epsilon\_{\mathbf W}$$
	则有：
	$$\mathbb E\_{p(\epsilon\_{\mathbf W})} [ (\tilde f(\mathbf{\vec x};\vec\theta,\epsilon\_{\mathbf W})-f(\mathbf{\vec x};\vec\theta))^{2}]=\mathbb E\_{p(\epsilon\_{\mathbf W})}(\nabla\_{\mathbf W}f(\mathbf{\vec x};\vec\theta) ^{T}\epsilon\_{\mathbf W})^{2}=||\nabla\_{\mathbf W}f(\mathbf{\vec x};\vec\theta)||^{2}\eta $$
	于是有：
	$$\tilde J(\vec\theta)= J(\vec\theta)+\frac {\eta }{2} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}[||\nabla\_{\mathbf W}f(\mathbf{\vec x};\vec\theta)||^{2}]  $$
	则得出结论：权重噪声注入的代价函数等于非权重噪声注入的代价函数加上一个正则化项。该正则化项就是 \\(\frac {\eta }{2} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}[||\nabla\_{\mathbf W}f(\mathbf{\vec x};\vec\theta)||^{2}  ]\\)，其中 \\(\eta\\) 为噪声的方差。
	- 该形式的正则化将鼓励参数进入对小扰动不敏感的区域。即：找到的点不仅是极小点，还是由平坦区域包围的极小点。
	> 平坦区域意味着梯度很小，意味着对小扰动不敏感。

3. 如果是简单的线性回归，即 \\(f(\mathbf{\vec x};\vec\theta)=\mathbf{\vec w}^{T}\mathbf{\vec x}+b\\)，则正则化项退化为 \\(\frac {\eta }{2} \mathbb E\_{\mathbf{\vec x},y\sim \hat p\_{data}}[|| \mathbf{\vec x}||^{2} ]\\)
	- 该正则化项与模型的参数无关，因此对  \\(\tilde J(\vec\theta)\\) 关于 \\(\mathbf W\\) 的梯度没有贡献。

4. 有些数据集的部分数目的 \\(y\\) 标签是错误的，此时通过最大似然准则来最大化 \\(\sum \log p(y\mid \mathbf{\vec x})\\) 是不正确的。一种解决方案是，显式地对标签上的噪音进行建模：	假设某些很小的常数 \\(\epsilon\\)， 标签 \\(y\\) 为正确的概率为 \\(1-\epsilon\\) 。

	如基于 \\(k\\) 个输出的`softmax`单元的标签平滑正则化`label smoothing regularize`：将输出值从 `0/1`替换为 \\(\frac{\epsilon}{k}/1-\frac{k-1}{k}\epsilon\\).
	> 本来 \\(k-1\\) 个输出为 0， 1个输出为 1；替换为：\\(k-1\\) 个输出为 \\(\frac{\epsilon}{k}\\) ， 1个输出为 \\(1-\frac{k-1}{k}\epsilon\\)

	通过最大似然函数训练`softmax`分类器时，训练过程可能永远不收敛：因为`softmax`单元不可能输出概率 `0`和概率`1`。解决的办法有两个
	- 采用 \\(L_{2}\\) 正则化策略来添加正则化项，而不仅仅是最大似然函数
	- 采用标签平滑正则化。此时当输出接近 `0/1`时即可认为是正确的。

## 七、 半监督学习与多任务学习
1. 在深度学习的背景下，半监督学习指的是学习一个表达`representation` ： \\(\mathbf{\vec h}=f(\mathbf{\vec x})\\)
	- 学习的目标是：使得同类中的样例有类似的表达
	
2. 我们可以构建一个这样的模型：生成模型 \\(P(\mathbf{\vec x})\\) （或者  \\(P(\mathbf{\vec x},y)\\)） 与判别模型  \\(P(y \mid \mathbf{\vec x})\\)  共享参数。其中：
	- 生成模型 \\(P(\mathbf{\vec x})\\) （或者  \\(P(\mathbf{\vec x},y)\\)） 表达了对于监督学习问题解的先验知识。即 \\(P(\mathbf{\vec x})\\) 的结构通过共享参数的方式连接到   \\(P(y \mid \mathbf{\vec x})\\) 
	- 此时不需要将无监督学习和监督学习部分进行分离
	- 此时对监督准则 \\(-\log P(y\mid \mathbf{\vec x})\\) 与无监督准则 \\(-\log P( \mathbf{\vec x})\\) （或者 \\(-\log P(\mathbf{\vec x},y)\\)） 进行权衡，这样我们可以得到比单纯的生成模型或者单纯的判别模型更好的模型，从而提高了泛化能力

3. 多任务学习是指通过几个任务中共享相同的样例。
	- 这可以视作一种参数上的软约束

4. 下图给出了多任务学习中的一个非常普遍的形式：有两个监督任务和一个无监督任务。
	- 所有的任务都共享相同的输入 \\(\mathbf{\vec x}\\) 和第一级中间层 \\(\mathbf{\vec h}^{(share)}\\)
	- 具体的任务使用了具体的表示层
	- 因为共享参数，其统计强度大大提高，因此能改进泛化能力。但是要求：确实有某些参数在不同任务之间共享了（这要求不同任务之间存在某些统计关系）

5. 从深度学习的观点看，多任务学习刻画了一个先验知识：这些不同的任务中，能解释数据变化的因子是跨任务共享的

	![mul_task](../imgs/7/mul_task.PNG)

## 八、 早停
### 1. 早停算法

1. 当训练一个有足够的表达能力的大模型时，我们会经常发现：训练误差会逐渐降低，但是验证集的误差先下降后上升！
	- 我们可以返回使得验证集误差最低的参数。因此我们每次需要存储模型参数；当训练算法终止时，返回的不是最新的参数，而是验证集误差最小的参数。

	![train_validate](../imgs/7/train_validate.PNG)

2. 当验证集上的误差没有进一步改善时，算法就提前终止。这种策略被称作早停`early stopping`。
	- 这可能是深度学习中最常用的正则化形式，因为它有效而且简单

3. 早停算法：
	- 输入：
		- 当前验证集的误差不是最小值的次数 \\(p\\)
		- 两次验证集上验证的间隔 \\(n\\)
		- 初始参数 \\(\vec\theta\_0\\)
	- 输出：
		- 最佳参数 \\(\vec\theta^{\*}\\)
		- 获得最佳参数时迭代的步数  \\(i^{\*}\\)
	- 算法步骤：
		- 初始化：
			- 参数变量 \\(\vec\theta=\vec\theta\_0\\)
			- 迭代步数变量 \\(i=0\\)
			- 验证次数变量 \\(j=0\\)
			- 验证集的最小误差 \\(v=\infty\\)
			- 最佳参数 \\(\vec\theta^{\*}=\vec\theta\\)
			- 最佳迭代步数 \\(i^{\*}=i\\)
		- 循环，循环条件为： \\(j \lt p\\)：
			- 学习模型 \\(n\\) 步
			- 更新 \\(i=i+n\\)
			- 记录最新的验证集误差 \\(v^{\prime}=\text{ValidationSetError}(\vec\theta)\\)
			- 如果 \\(v^{\prime} \lt v\\)，则：
				$$j=0,\vec\theta^{\*}=\vec\theta,i^{\*}=i,v=v^{\prime}		$$
			如果 \\(v^{\prime} \ge v\\)，则 \\(j=j+1\\)

4. 我们可以认为：早停是一个非常高效的超参数选择算法
	- 训练步数是一个超参数。该超参数在验证集上具有 `U`形曲线
	- 早停策略通过拟合训练步数来控制模型的有效容量`capacity`
	- 早停策略只需要跑一轮训练就能够得到很多的超参数（指的是训练步数这个超参数），

5. 早停策略的代价有两个：
	 - 一是需要在训练期间定期评估验证集
		- 可以通过并行的执行训练和验证来加速这一过程
		- 也可以选取一个较小的验证集、或者不那么频繁地评估验证集来减小评估的代价
	- 二是需要保持最佳的参数的副本。这种代价一般可以忽略不计。

6. 早停是正则化的一种非常不显眼的形式
	- 它几乎不需要干涉基本训练过程、目标函数、参数的取值的约束
	- 可以单独使用，或者与其他的正则化策略相结合
	- 早停不仅带来了正则化的好处，还带来了降低计算成本的好处

7. 早停需要验证集，这意味着某些样本不能用于模型的训练过程。为了更好地利用验证集，我们可以在提前终止初步训练之后，进行额外的训练
	- 在第二轮额外的训练中，所有的训练数据都被包括在内（包括验证集）
	- 有两个基本的策略可以用于第二轮训练过程 
	
	一种策略是：再次初始化模型，然后使用所有数据再次训练。此时：我们使用第一轮早停确定的最佳步数作为第二轮的迭代步数

	- 输入：训练集 \\(\mathbf X^{(train)}\\) ，  \\(\mathbf y^{(train)}\\)  
	- 步骤：
		- 将 \\(\mathbf X^{(train)}\\) 和 \\(\mathbf y^{(train)}\\) 分割为  \\((\mathbf X^{(subtrain)},\mathbf y^{(subtrain)})\\) 和 \\((\mathbf X^{(valid)},\mathbf y^{(valid)})\\)
		- 随机选择参数的初始化值 \\(\vec\theta_0\\)，将 \\((\mathbf X^{(subtrain)},\mathbf y^{(subtrain)})\\) 作为训练集， 将 \\((\mathbf X^{(valid)},\mathbf y^{(valid)})\\) 作为验证集，运行早停算法，返回最佳训练步数 \\(i^{\*}\\)
		- 再次选择参数的另一个初始化值 \\(\vec\theta_1\\)，在 \\((\mathbf X^{(train)},\mathbf y^{(train)})\\) 上再次训练  \\(i^{\*}\\) 步


	另一个策略是：保持从第一轮训练中获得的参数，然后使用全部的数据继续训练。此时我们观察验证集的损失函数（根据早停策略，第二轮初始时，验证集的损失函数值是较大的），直到它低于第一轮停止时的训练集的损失函数值（这个值是较小的）。
	> 这个策略避免了重新训练模型的高成本，但是表现一般

	- 输入：训练集 \\(\mathbf X^{(train)}\\) ，  \\(\mathbf y^{(train)}\\)  
	- 步骤：
		- 将 \\(\mathbf X^{(train)}\\) 和 \\(\mathbf y^{(train)}\\) 分割为  \\((\mathbf X^{(subtrain)},\mathbf y^{(subtrain)})\\) 和 \\((\mathbf X^{(valid)},\mathbf y^{(valid)})\\)
		- 随机选择参数的初始化值 \\(\vec\theta_0\\)，将 \\((\mathbf X^{(subtrain)},\mathbf y^{(subtrain)})\\) 作为训练集， 将 \\((\mathbf X^{(valid)},\mathbf y^{(valid)})\\) 作为验证集，运行早停算法，返回算法停止时的目标函数的值 \\(\epsilon=J(\vec\theta,\mathbf X^{(subtrain)},\mathbf y^{(subtrain)})\\)
		- 迭代： \\( while \;J(\vec\theta,\mathbf X^{(valid)},\mathbf y^{(valid)}) \gt \epsilon\quad do\\)
			- 在 \\(\mathbf X^{(valid)},\mathbf y^{(valid)}\\) 上训练 \\(n\\) 步
			> 每隔 \\(n\\) 步检查一次

###2. 早停与 L2 正则化
1. 早停充当正则化的原理：早停将优化过程的参数空间限制在初始参数值 \\(\vec\theta_0\\) 的一个小的邻域内。

	假设用学习率 \\(\epsilon\\) 进行了 \\(\tau\\) 个优化步骤（对应于 \\(\tau\\) 个训练迭代）：
	- 我们将 \\(\epsilon\tau\\) 视作模型的有效容量 `effective capacity` 的度量。
	- 假设梯度有界，则限制迭代的次数和学习率，会限制 \\(\vec\theta\\) 从 \\(\vec\theta_0\\) 能够到达的范围。
	> 此时 \\(\epsilon\tau\\) 的行为就像是权重衰减系数的倒数。权重系数越大，则 \\(\vec\theta\\) 越小，等价于  \\(\epsilon\tau\\)  越小。

	下图为早停与 \\(L_2\\) 正则化的示意图：

	![early_stopping](../imgs/7/early_stopping.PNG)

	- 左图/右图实线：负的对数似然函数（即 \\(J\\) ) 的等值线。最小点是 \\(\mathbf{\vec w}^{\*}\\) 
	- 左图虚线：从原点开始的 `SGD` 经过的轨迹。
	- 右图虚线为 \\(L_2\\) 正则化项的等值线。

	由于早停，轨迹在 \\(\tilde{\mathbf{\vec w}}\\) 处停止，而不是在 \\(\mathbf{\vec w}^{\*}\\)  点。 而 \\(L\_2\\) 的正则化项使得总代价函数的最小值比非正则化代价函数的最小值更靠近原点。

2.  假设参数 \\(\vec\theta=\mathbf{\vec w}\\)。令 \\(\mathbf{\vec w}^{\*}=\arg\min\_{\mathbf{\vec w}}J(\mathbf{\vec w})\\)，它就是无正则化项时使得目标函数最小的权重向量。和 \\(L_2\\) 正则化中的推导相同，有：
	$$\nabla\_{\mathbf{\vec w}} \hat J(\mathbf{\vec w})=\mathbf H(\mathbf{\vec w}-\mathbf{\vec w}^{\*})$$
		\\(\mathbf H\\) 为 \\(J\\) 在  \\(\mathbf{\vec w}^{\*}\\) 处的海森矩阵（由于 \\(\mathbf{\vec w}^{\*}\\) 是 \\(J(\mathbf{\vec w})\\) 的最小点，因此海森矩阵 \\(\mathbf H\\) 是半正定的）。 其中 \\(\mathbf{\vec w}\\)  在  \\(\mathbf{\vec w}^{\*}\\) 的附近。 

	迭代过程为：
	$$\mathbf{\vec w}^{(\tau)}=\mathbf{\vec w}^{(\tau-1)}-\epsilon \nabla\_{\mathbf{\vec w}}\hat J(\mathbf{\vec w}^{(\tau-1)})=\mathbf{\vec w}^{(\tau-1)}-\epsilon\mathbf H(\mathbf{\vec w}^{(\tau-1)}-\mathbf{\vec w}^{\*})\\\
	\rightarrow \mathbf{\vec w}^{(\tau)}-\mathbf{\vec w}^{\*}=(\mathbf I-\epsilon\mathbf H)(\mathbf{\vec w}^{(\tau-1)}-\mathbf{\vec w}^{\*})$$
	>这里并没有加入任何正则化项，而是使用 \\(J\\)	

	 将 \\(\mathbf H\\) 进行特征分解： \\(\mathbf H=\mathbf Q\Lambda\mathbf Q^{T}\\) ，其中 \\(\Lambda\\) 为对角矩阵， \\(\mathbf Q\\) 为特征向量的一组标准正交基。则有：
	$$\mathbf{\vec w}^{(\tau)}-\mathbf{\vec w}^{\*}=\mathbf Q(\mathbf I-\epsilon\Lambda)\mathbf Q^{T}(\mathbf{\vec w}^{(\tau-1)}-\mathbf{\vec w}^{\*}) \\\
\rightarrow  \mathbf Q^{T}(\mathbf{\vec w}^{(\tau)}-\mathbf{\vec w}^{\*})=(\mathbf I-\epsilon\Lambda)\mathbf Q^{T}(\mathbf{\vec w}^{(\tau-1)}-\mathbf{\vec w}^{\*})
	$$

	我们令参数向量的初始值为原点： \\(\mathbf{\vec w}^{(0)}=\mathbf{\vec 0}\\) 。并且选择 \\(\epsilon\\) 使得 \\(|1-\epsilon\lambda_i|\lt 1\\)（\\(\lambda_i\\) 为 \\(\mathbf H\\) 的特征值）。则经过 \\(\tau\\) 次参数更新之后：
	$$ \mathbf Q^{T}\mathbf{\vec w}^{(\tau)}=[\mathbf I-(\mathbf I-\epsilon\Lambda)^{\tau}]\mathbf Q^{T}\mathbf{\vec w}^{\*}$$

	而在 \\(L_2\\) 正则化项中，有：
	$$\tilde{\mathbf{\vec w}}^{\*}=\mathbf Q(\mathbf  \Lambda+ \alpha\mathbf I)^{-1}\mathbf  \Lambda\mathbf Q^{T} \mathbf{\vec w}^{\*} \\\
	\rightarrow \mathbf Q^{T}\tilde{\mathbf{\vec w}}^{\*}=(\mathbf  \Lambda+ \alpha\mathbf I)^{-1}\mathbf  \Lambda\mathbf Q^{T} \mathbf{\vec w}^{\*}\\\
	= [\mathbf I-(\Lambda+\alpha\mathbf I)^{-1}\alpha]\mathbf Q^{T} \mathbf{\vec w}^{\*}
		$$
	> 通过直接写出逆矩阵的形式可以证明

3. 如果超参数 \\(\epsilon,\alpha,\tau\\) 满足：
	$$(\mathbf I-\epsilon\Lambda)^{\tau}=(\Lambda+\alpha\mathbf I)^{-1}\alpha  $$
	则：\\(L_2\\) 正则化等价于早停。

	为了求得三个超参数满足的条件，我们分析：
	$$
	\begin{bmatrix}
	(1-\epsilon\lambda_1)^{\tau}&0&\cdots&0\\\
	0&(1-\epsilon\lambda_2)^{\tau}&\cdots&0\\\
	\vdots&\vdots&\ddots&\vdots\\\
	0&0&\cdots&(1-\epsilon\lambda_n)^{\tau}\\\
	\end{bmatrix}=\begin{bmatrix}
	\frac{\alpha}{\lambda_2+\alpha}&0&\cdots&0\\\
	0&\frac{\alpha}{\lambda_1+\alpha}&\cdots&0\\\
	\vdots&\vdots&\ddots&\vdots\\\
	0&0&\cdots&\frac{\alpha}{\lambda_n+\alpha}\\\
	\end{bmatrix}
	$$
	即有：
	$$(1-\epsilon\lambda_i)^{\tau}=\frac{\alpha}{\lambda_i+\alpha} $$
	两边取对数，然后使用 \\(\log(1+x)\\) 的级数展开有（假设 \\(0 \lt\epsilon\lambda_i \ll 1\\)，且 \\(\frac{\lambda_i}{\alpha} \ll 1\\)）：
	$$\tau\epsilon \lambda_i=\frac{\lambda_i}{\alpha}$$
	则有：
	$$\tau\epsilon=\frac{1}{\alpha}$$
	说明了： \\(\tau\epsilon\\) 的倒数与 \\(L_2\\) 的权重衰减系数的作用类似。

4. 早停比 \\(L_2\\) 正则化更有优势：
	- 早停能够监控验证集的误差，从而自动在某个较好的参数解的位置终止
	- 采用 \\(L_2\\) 正则化需要多次训练，从而选择合适的超参数 \\(\alpha\\) 的值

##九、参数共享
1. 通常我们对参数添加约束时，是固定于相对的某个点，如 \\(L_2\\) 正则化：将参数对于偏离零点来进行惩罚。
	- 如果我们需要表达两个模型之间参数的相对关系时，则使用参数共享。

2. 参数的相对约束：假设模型 `A`的参数为 \\(\mathbf{\vec w}^{(A)}\\)，模型 `B`的参数为 \\(\mathbf{\vec w}^{(B)}\\) 。如果两个模型非常相似，则我们给定下列形式的惩罚：
	$$\Omega(\mathbf{\vec w}^{(A)},\mathbf{\vec w}^{(B)})=||\mathbf{\vec w}^{(A)}-\mathbf{\vec w}^{(B)} ||\_2^{2}$$
	这里使用 \\(L_2\\) 惩罚。你可以使用其他形式的正则化形式。

	这种方法由`Lasserre et al.(2006)`提出，它使得一个有监督的分类模型的参数接近于另一个无监督的输入数据的分布模型的参数。

3. 还有一种方案是：使用约束：强迫 \\(\mathbf{\vec w}\\) 的某个子集相等。这称作参数共享`parameter sharing`。
	- 它的优点是：共享的参数子集只需要存储一份在内存中。对于某些特定模型（如卷积神经网络），他显著减少模型占用的内存

## 十、 稀疏表达
1. 有一种策略是：鼓励 \\(\mathbf{\vec x}\\) 的表达 \\(\mathbf{\vec h}\\) 为稀疏的。即：
	$$\tilde J(\vec\theta;\mathbf X,\mathbf{\vec y})=J(\vec\theta;\mathbf X,\mathbf{\vec y})+\alpha\Omega(\mathbf{\vec h}) $$
	它与常规正则化区别是：它对 \\(\mathbf{\vec h}\\) 进行限制，而不是 \\(\vec\theta\\) 进行限制

2. 要使得 \\(\mathbf{\vec h}\\)  是稀疏的，则有 \\(\Omega\\) 可以为：
	- 采用 \\(L_1\\) 正则化：  \\(\Omega(\mathbf{\vec h})=||\mathbf{\vec h}||_1\\) 

3. 还有其他方法可以获取稀疏表达。如正交匹配追踪`orthogonal matching pursuit`：
	$$ \arg\min\_{\mathbf{\vec {h}},||\mathbf{\vec {h}}||\_0 \lt k}||\mathbf{\vec x}-\mathbf W\mathbf{\vec h}||^{2}$$
	其中 \\(||\mathbf{\vec {h}}||\_0\\) 是 \\( \mathbf{\vec {h}} \\) 中非零项的个数。当  \\(\mathbf W\\) 被限定为正交时，该问题可以被高效解决。
	
## 十一、 Bagging 和其他集成的方法
1. `bagging`通过结合几个模型来降低泛化误差：
	- 分别训练几个不同的模型
	- 让所有模型表决来给出测试样本的输出

	这称作模型平均`model averaging`

2. 模型平均有效的原因是：不同的模型通常不会在测试集上产生完全相同的错误。

3. 给出 \\(k\\) 个回归模型，假设每个模型在样本上的误差是随机变量 \\(\epsilon_i.i=1,2,\cdots,k\\)。假设该随机变量服从均值为零、方差为 \\(\mathbb E[\epsilon_i^{2}]=v\\)、协方差为 \\(\mathbb E[\epsilon_i\epsilon_j]=c\\) 的多维正态分布。

	所有模型的平均预测所得的误差为 \\(\frac 1k\sum\_i\epsilon_i\\) 。则集成预测器的平方误差的期望为：
	$$ \mathbb E\left[\left(\frac 1k\sum\_{i}\epsilon_i\right)^{2}\right]=\frac {1}{k^{2}}\mathbb E\left[\sum\_{i}\left(\epsilon_i^{2}+\sum\_{j\ne i}\epsilon_i\epsilon_j\right)\right] =\frac 1k v+\frac{k-1}{k}c$$
	当误差完全相关时，（即 \\(c=v\\)），则集成预测器的均方误差为 \\(v\\) ，没有任何帮助；当误差完全不想管时，（即 \\(c=0\\)），则集成预测器的均方误差为 \\(\frac vk  \\)。
	- 集成平均模型至少与它的任何成员表现得一样好
	- 如果成员之间的误差是独立的，则集成模型显著地比它的成员表现得更好

4. 不同的集成方法使用不同的方式构建集成模型
	- 集成模型的每个成员可以使用不同算法和不同目标函数
	- 一种典型方式是： `bagging` 方法

5. `bagging`方法：
	- 构造 \\(k\\) 个不同的数据集，每个数据集与原始数据集具有相同数量的样本，通过对原始数据集进行有放回的重复采样得到。
	- 模型 \\(i\\) 在数据集 \\(i\\) 上进行训练。每个数据集包含的样本的差异将会导致模型之间的差异

6. 在神经网络中，由于随机初始化的差异、`minibatch`的随机选择、超参数的差异、不同的输出的实现方式，将导致足够多的变化，从而使得神经网络可以从模型平均中受益

7. 由于任何机器学习算法都可以从模型平均中大幅获益，因此这种方法不适合作为评价算法的基准模型。
	
8. 模型平均的代价是：计算量增加、存储量增加

## 十二、 dropout

### 1. dropout 与 bagging

1. `dropout`可以视作集成了非常多的大型神经网络的`bagging`集成模型
	- 由于`bagging`涉及训练多个模型，并且在每个测试样本上评估多个模型。当模型是一个大型神经网络时，这种`bagging`计算量非常庞大，实际不可行
	- `dropout`提供了一种方便的`bagging`近似，它能够训练和评估指数级别的神经网络。

2. `dropout`训练的集成模型包含了：所有从基本的基础网络中删除非输出单元形成的子网络。
	- 被删除的单元可以是隐单元，也可以是输入单元

	下图中，右侧给出了`dropout`使用的所有子网络。
	
	![dropout](../imgs/7/dropout.PNG)

3. 假定一个掩码向量 \\(\vec \mu\\)，他给出了哪些单元被保留哪些单元被删除（掩码为 0 的单元被删除）。

	定义 \\(J(\vec\theta,\vec\mu)\\) 为参数 \\(\vec\theta\\) 和掩码 \\(\vec\mu\\) 定义的模型代价。`dropout`的目标是最小化 \\(\mathbb E\_{\vec\mu}J(\vec\theta,\vec\mu)\\)
	- 期望包含了指数多项（因为掩码向量有指数多个），但是我们可以通过抽样 \\(\vec\mu\\) 来获得梯度的无偏估计

4. `dropout`训练与`bagging`训练不同：
	- `bagging`中，假设所有的子模型都是独立的
	- `dropout`中，所有的子模型是共享参数的。每个子模型继承了基础神经网络的不同子集。参数共享使得在有限的内存下训练指数数量的子模型变得可能。
	- `bagging`中，每个子模型在其相应的训练集上训练到收敛
	- `dropout`中，大部分子模型都没有显式的被训练（因为指数量级的神经网络不可能被训练完成）。我们只是对子网络的某些部分训练单个步骤，同时参数共享使得剩余的子网络能够有较好的参数设定。

	除此之外，二者相同。比如每个子网络使用的训练集是原始训练集的一个有放回重复采样而来。

5. 我们假设`bagging`模型的作用是输出一个概率分布（这个过程称作推断`inference`）。

	在`bagging`的情况下，每个子模型 \\(i\\) 产生一个概率分布 \\(p^{(i)}(y\mid \mathbf{\vec x})\\)，集成模型的预测由这些分布的算术平均给出：
	$$\frac 1k\sum\_{i=1}^{k} p^{(i)}(y\mid \mathbf{\vec x})$$

6. 在`dropout`情况下，通过掩码 \\(\vec{\mu}\\) 定义了每个子模型的概率分布 \\(p(y\mid \mathbf{\vec x},\vec \mu)\\)
	> 这里并没有给出 \\(k\\) 个不同的子模型，而是根据不同的 \\(\vec{\mu}\\) 产生不同的子模型。

	集成模型的预测由所有掩码的算术平均值给出：
	$$\sum\_{\vec\mu}p(\vec\mu)p(y\mid \mathbf{\vec x},\vec \mu) $$
	其中 \\(p(\vec\mu)\\) 是训练时， \\(\vec\mu\\) 的概率分布。上式包含了指数量级的项，不可能计算出来。

	- 我们可以通过采样来近似推断，即平均多个掩码对应的输出。通常 10-20 个掩码就足以获取不错的表现。
	- 还有一个更好的方法能获取一个不错的近似预测：我们采用几何平均而不是算术平均：
	$$ \tilde p\_{ensemble}(y\mid \mathbf{\vec x})=\sqrt[2^{d}]{\prod\_{\vec \mu}p(y\mid \mathbf{\vec x},\vec \mu) }$$
	其中 \\(d\\) 为所有的可以被丢弃的单元的数量。这里我们采用的是均匀分布的 \\(\vec\mu\\)（你也可以采用非均匀分布）

	多个概率分布的几何平均不一定是个概率分布。因此为了保证结果是一个概率分布，我们要求：
	- 没有任何一个子模型给出某个事件的概率为 0
	- 概率归一化为：
	$$  p\_{ensemble}(y\mid \mathbf{\vec x})=\frac{\tilde  p\_{ensemble}(y\mid \mathbf{\vec x})}{\sum\_{y^{\prime}} \tilde p\_{ensemble}(y^{\prime}\mid \mathbf{\vec x})}$$

7. `dropout`一个重要观点是：我们可以通过评估某个模型的  \\( p (y\mid \mathbf{\vec x}) \\) 来近似 \\( p\_{ensemble} \\) 。这个模型这样产生：
	- 该模型与基础模型具有相同的单元
	- 该模型单元 \\(i\\) 输出的权重需要乘以包含该单元 \\(i\\) 的概率

	这种策略叫做权重比例推断法则。目前这种做法在深度非线性网络上工作良好，但是没有任何理论上的说法。

	通常我们使用 \\(\frac 12\\) 的包含概率，即：近半比例的权重在训练时丢失。

### 2. dropout 示例

1. 考虑`softmax`分类，假设 \\(\mathbf{\vec v}\\) 表示 \\(n\\) 个输入变量，则有：
	$$p(\mathbf y=y\mid \mathbf{\vec v})=softmax(\mathbf W^{T}\mathbf{\vec v}+\mathbf{\vec b})\_y $$

	我们根据二值向量 \\(\mathbf{\vec d}\\) 逐元素的乘法，给出：
	$$p(\mathbf y=y\mid \mathbf{\vec v};\mathbf{\vec d})=softmax(\mathbf W^{T}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b})\_y  $$
	其中 \\(\odot\\) 表示逐元素的相乘。

	集成预测器的预测为：
	$$  p\_{ensemble}(\mathbf y=y\mid \mathbf{\vec v})=\frac{\tilde  p\_{ensemble}(\mathbf y=y\mid \mathbf{\vec v})}{\sum\_{y^{\prime}} \tilde p\_{ensemble}(\mathbf y=y^{\prime}\mid \mathbf{\vec v})}$$
	其中 

	$$ \tilde p\_{ensemble}(\mathbf y=y\mid \mathbf{\vec v})=\sqrt[2^{n}]{\prod\_{\mathbf{\vec d}\in \\{0,1\\}^{n}}p(\mathbf y=y\mid \mathbf{\vec v};\mathbf{\vec d}) }$$

	化简上式：
	$$\tilde p\_{ensemble}(\mathbf y=y\mid \mathbf{\vec v})=\sqrt[2^{n}]{\prod\_{\mathbf{\vec d}\in \\{0,1\\}^{n}}softmax(\mathbf W^{T}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b})\_y  }\\\
	= \sqrt[2^{n}]{\prod\_{\mathbf{\vec d}\in \\{0,1\\}^{n}}\frac{\exp(\mathbf W^{T}\_{y,:}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b}\_y)}{\sum\_{y^{\prime}}\exp(\mathbf W^{T}\_{y^{\prime},:}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b}\_{y^{\prime}})} }$$

	忽略对 \\(y\\) 不变的项，有：
	$$ \tilde p\_{ensemble}(\mathbf y=y\mid \mathbf{\vec v}) \propto \sqrt[2^{n}]{\prod\_{\mathbf{\vec d}\in \\{0,1\\}^{n}}  \exp(\mathbf W^{T}\_{y,:}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b}\_y)} \\\
	= \exp\left(\frac 1{2^{n}}\sum\_{\mathbf{\vec d}\in \\{0,1\\}^{n}}  \mathbf W^{T}\_{y,:}(\mathbf{\vec v}\odot \mathbf{\vec d})+\mathbf{\vec b}\_y \right)\\\
	=\exp\left(\frac 12\mathbf W\_{y,:}\mathbf{\vec v}+\mathbf{\vec b}\_y\right) $$
	> 最后一步是这样考虑： \\(\sum\\) 一共有 \\(2^{n}\\) 个组合。在其中 \\(d_i\\) 取 0/1 的概率都是 \\(\frac 12\\)，因此其期望就是最后一个结果。

	因此我们得到：`softmax`分类器采用`dropout`的集成分类器就是权重为 \\(\frac 12\mathbf W\\) 的`softmax`分类器。

2. 在神经网络中， \\(\mathbf{\vec v}\\) 其实就是输出层对应的输入 \\(\mathbf{\vec h}\\)，并不是整个网络的输入 \\(\mathbf{\vec x}\\) 
 
### 3. dropout 性能
1. 权重比例推断法制在其他情况下也是精确的，但是它对于非线性的深度模型仅仅是一个近似
	- 虽然尚未有理论上的分析，但是在实践中往往效果非常好。
	- 实践表明，它比蒙特卡罗模拟工作得更好

2. `Srivastava et al.(20140`显示：`dropout`比其他标准的计算开销较小的正则化项（如 \\(L_2\\) 正则化）更加有效
	- `dropout`也可以与其他形式的正则化合并，进一步提升预测性能

3. `drouput`的一个显著优点是：计算非常方便。
	- 训练过程中，使用`dropout`产生 \\(n\\) 个随机二进制数与每个权重相乘。每个样本每次更新只需要 \\(O(n)\\) 的计算复杂度。根据需要，也可能需要 \\(O(n)\\) 的存储空间来保存这些二进制数（直到反向传播阶段）
	> 训练过程是这样的：与基础模型一样的训练步骤。但是每次更新一个样本时，根据 \\(n\\) 个随机二进制数来选择留下来的单元，而不是每个单元都留下来

	- 使用训练好的模型推断时，计算未知样本的代价与不使用`dropout`是一样的（但是我们在开始推断之前，必须将权重除以 2）

4. `dropout`另一个显著优点是：不会限制适用的模型或者训练过程。
	- 其他某些效果较好的正则化策略对模型结构有严格的限制

5. 虽然`dropout`在模型的每一步上所做的工作是微不足道的（仅仅是随机地保留某些单元），但是在完整的系统上，该策略的结果是非常显著的。
	- `dropout`是一个正则化技术，它减少了模型的有效容量`effective capacity`
	- 为了抵抗模型有效容量的降低，我们必须增大模型的规模。此时需要更多的训练迭代次数。
	- 对于非常大的数据集，`dropout`正则化带来的泛化误差减少得很小。此时使用`dropout`带来的更大模型的计算代价可能超过了正则化带来的好处
	- 当只有极少的训练样本可用时，`dropout`不会很有效
	- 当有其它未分类的数据可用时，无监督特征学习比`dropout`更有效
6. `Wager et al.(2013)`表明：当`dropout`作用于线性回归时，相当于每个输入特征具有不同权重的 \\(L_2\\) 正则化。
	- 每个特征的权重是由其方差来确定

	对于其他线性模型也有类似结果。但是对于深度模型，`dropout`与 \\(L_2\\) 正则化是不等同的。

7. `dropout`训练时的随机选择保留哪些单元（输出单元一定全部保留）。这种随机选择并不是产生正则化效果的必要条件，它仅仅是近似所有可能的子模型总和的一种方法。
	- `Wang and Manning(2013)`导出了近似这种边缘分布的解析解，这种近似被称作快速`dropout`(`fast dropout`)。
		- 这种方法使得梯度计算中的随机性减小，从而获得了更快的收敛时间
		- 这种方法在小的神经网络上的性能几乎与标准的`dropout`相当，但是在大的模型上，尚未产生显著改善

8. `dropout`训练时采用的随机性也不是产生正则化效果的充分条件。`Warde-Farley et al.(2014)`使用 `Dropout Boosting`策略（类似于`boosting`）设计了一个对照试验，而传统的`dropout`方法类似于`bagging`。
	- 结果表明`dropout boosting`几乎没有正则化的效果
	- 当随机抽样的个体子模型相互独立训练好之后，`bagging`集成的正则化效果才能显现
	>- 传统的`bagging`方法：随机性->多个子模型->组合成最终的模型。这里多个子模型是独立训练的。
	>- `dropout boosting`方法：随机性->多个子模型->组合成最终的模型。这里多个子模型是采用 boost 策略训练的。下一个子模型拟合上一个子模型的残差。

9. `dropout`启发了人们产生一类策略：通过以随机方法来训练那些有着指数量级子模型的共享权重的集成方法。
	- `dropconnect`是`dropout`的一个特殊情况：一个标量权重和单个隐单元状态之间的每个乘积被认为是可丢弃的一个单元
	- 随机池化是构造卷积神经网络集成方法的一种策略
	- 目前为止，`dropout`仍然是最广泛使用的隐式集成方法

10. `dropout`是通过随机行为训练网络，并平均了多个随机决定来预测，然后通过参数共享实现了`bagging`
	- 这种参数共享策略不一定必须给予包括和排除（也就是二进制的 0 和1）。原则上，任何一种随机的修改都可以接受。
	- 我们可以认为随机向量 \\(\vec\mu\\) 具有任何形式，而不一定要求它是掩码向量（取值为 0/1）。`Srivastava et al.(2014)`表明，将权重乘以 \\(\vec\mu\sim \mathcal (\mathbf{\vec 1},\mathbf I)\\) 要比基于二值掩码`dropout`表现的更好
		- 由于此时 \\(\mathbb E[\vec\mu]=1\\)，则网络自动实现了集成方法的近似推断，则不需要应用权重比例推断规则

11. `dropout`不仅仅是一种高效的近似`bagging`的方法，它还是共享隐单元的集成模型
	- 无论隐单元是否在模型中，每个隐单元必须能够表现良好
	- 隐单元必须时刻准备好进行子模型之间的交换和互换

12. `dropout`强大的大部分是由于施加到隐单元的掩码噪声。
	- 这可以看作是对输入内容的信息自适应破坏的一种形式，而不是简单地对输入原始值的破坏。它是对输入内容的某个结构进行了修改，而不是传统的噪声注入。
	- `dropout`的噪声是乘性的，而不是加性的。

## 十三、对抗训练
1. 神经网络在预测过程中，很有可能出现这样的现象：对输入点 \\(\mathbf{\vec x}\\) 和该点附近的 \\(\mathbf{\vec x}^{\prime}\\) 给出完全不同的预测。
	- 输入点 \\(\mathbf{\vec x}\\) 来自于训练集，称作原始样本；而  \\(\mathbf{\vec x}^{\prime}\\)  可以是人工构造的，称作对抗样本`adversarial example`
	- 我们可以通过将对抗样本加入训练集来改善模型的泛化能力。这称作对抗训练`adversarial training`

2. `Goodfellow et al(2014b)`表明，这些对抗样本的主要原因是：过度线性。
	- 神经网络主要基于线性块构建的，因此模型是高度线性的。
	- 对于一个线性函数，如果它有多个输入，那么其函数值可以迅速改变。如果对每个输入添加一个扰动 \\(\epsilon\\)，则权重为 \\(\mathbf{\vec w}\\) 的线性函数的变化幅度为 \\(\epsilon||\mathbf{\vec w}||\_1\\)。如果 \\(\mathbf{\vec w}\\) 是高维的，则这将是一个非常大的数
	
3. 对抗训练通过鼓励网络在训练数据附近的局部区域保持稳定来限制函数的高度敏感性。
	- 它可以视作一种先验知识：模型是局部稳定的

4. 对抗训练也可以用于实现半监督学习：
	- 首先根据规模小的、监督样本中学习模型
	- 然后根据模型，预测大量的、未监督的样本。假设未监督样本 \\(\mathbf{\vec x}\\) 预测的标签为 \\(\hat y\\)（虽然其真实标签可能不是 \\(\hat y\\)，但是如果模型质量非常好，则真实标签是 \\(\hat y\\) 的概率非常大）
	- 然后在 \\(\mathbf{\vec x}\\) 的一个很小的领域内它的对抗样本 \\(\mathbf{\vec x}^{\prime}\\)。由于不是采用真实的标签，而是模型预测的标签 \\(\hat y\\)，因此  \\(\mathbf{\vec x}^{\prime}\\) 也称作虚拟对抗样本。
	- 重新训练模型，使得模型在 \\(\mathbf{\vec x},\mathbf{\vec x}^{\prime}\\)  上的预测标签分类都相同。

	这种策略鼓励模型沿着未监督样本所在流形上任意微小变化都是稳定的。其基本假设是：不同的分类通常位于分离的流形上，并且小的扰动不能从一类的流形跳到另一个流形上。

## 十四、切面距离、正切传播、流形正切分类器
1. 切面距离算法：最近邻算法的一种，其中的度量使用的不是通用的欧几里得距离。
	- 假设我们的分类样本有这样的性质：在同一个流形上的样本分类相同；不同流形上的样本分类不同。
	- 则我们的分类器有这样的特点：对于样本的局部变化不敏感（即：样本在流形上的移动时，分类器的输出不变）
	- 假设样本 \\(\mathbf{\vec x}\_1\\) 的流形为 \\(M_1\\)，样本 \\(\mathbf{\vec x}\_2\\) 的流形为 \\(M_2\\)。则这两个点的距离定义为：流形 \\(M_1\\) 到流形 \\(M_2\\) 的距离。
	- 但是问题是：计算两个流形的距离代价很高。它要求解两个集合的最近点。
	- 一个替代方案是：用 \\(\mathbf{\vec x}\_i\\) 点的切平面来近似 \\(M_i\\)。 然后测量两个切平面的距离（或者一个切平面到一个点的距离）

2. 受切面距离算法的启发，正切传播算法训练神经网络分类器的策略是：使得神经网络的输出 \\(f(\mathbf{\vec x})\\) 沿着同样类别样本所在的流形上移动时，局部不变。
	- 局部不变性，要求  \\(f(\mathbf{\vec x})\\) 沿着流形的方向变化很小。也就是 \\(\nabla\_{\mathbf{\vec x}}f(\mathbf{\vec x})\\) 在流形的切向的分量为零。
	- 对于样本点 \\(\mathbf{\vec x}\_i\\)，其流形切向量为 \\(\mathbf{\vec v}\_i\\)。则局部不变性要求  \\(\nabla\_{\mathbf{\vec x}}f(\mathbf{\vec x}\_i)\\) 与  \\(\mathbf{\vec v}\_i\\) 正交。
	- 正切传播算法的正则化惩罚项 \\(\Omega\\) 要求 \\(f\\) 在 \\(\mathbf{\vec x}\\) 的 \\(\mathbf{\vec v}\\) 方向的导数是较小的：
	$$\Omega(f)=\sum\_i\left(\nabla\_{\mathbf{\vec x}}f(\mathbf{\vec x}\_i)^{T}\mathbf{\vec v}\_i\right)^{2} $$
	>- 这里并没有严格要求其为零，而是比较小即可。
	>- \\(\nabla\_{\mathbf{\vec x}}f(\mathbf{\vec x})\\) 是这样算出的：首先根据样本的类别来分成不同的流形。然后找出 \\(\mathbf{\vec x}\\) 在本流形的近邻点来计算出梯度。

3. 正切传播有两个主要缺点：
	- 模型的正则化只能抵抗无穷小的扰动。作为对比：显式的数据集增强能够抵抗较大扰动
	- 梯度沿着切向方向分量很小的假设对于采用了`relu`单元的模型是困难的。这类模型只能通过关闭单元或者缩小权重才能缩小其导数（因为 `relu`单元的导数与 \\(\mathbf{\vec w}\\) 无关）
	> `sigmoid`或者`tanh`单元就没有这个问题。因为可以通过采用大权重在饱和区来收缩导数。

4. 正切传播和人工转换的数据集增强都要求模型对于输入变化的某些特定的方向是不变的。而对抗训练要求模型对于输入的所有方向上的小变化都是不变的。

5. 流形正切分类器：
	- 使用自编码器通过无监督学习来学习流行的结构
	- 使用正切传播算法来正则化神经网络分类器