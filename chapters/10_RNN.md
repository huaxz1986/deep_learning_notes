# 十 序列建模：RNN

1. 循环神经网络`recurrent neural network:RNN` ：一类用于处理序列数据 $\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(\tau)}$ 的神经网络

   - 循环神经网络可以扩展到很长的序列，也可以处理可变长度的序列

   > 卷积神经网络专门处理网格化数据 $\mathbf X$ （如一个图形），它可以扩展到具有很大宽度和高度的图像。

2. 循环神经网络是一种共享参数的网络：参数在每个时间点上共享

   - 传统的前馈神经网络在每个时间点上分配一个独立的参数，因此网络需要学习每个时间点上的规则（即参数）
   - 循环神经网络在每个时间点上共享相同的权重

## 1. 展开计算图

1. 考虑动态系统的经典形式
   $$
   \mathbf{\vec s}^{(t)}=f(\mathbf{\vec s}^{(t-1)};\mathbf{\vec \theta})
   $$
   其中 $\mathbf{\vec s}^{(t)}$ 称作系统的状态。对于有限的时间步 $\tau$ ，应用 $\tau-1$ 次定义可以展开这个图：
   $$
   \mathbf{\vec s}^{(\tau)}=f(\mathbf{\vec s}^{(\tau-1)};\mathbf{\vec \theta})=\cdots=f(\cdots f(\mathbf{\vec s}^{(1)};\mathbf{\vec \theta})\cdots ;\mathbf{\vec \theta})
   $$
   以这种方式重复应用定义、展开等式，就得到不涉及循环的表达式。现在可以用传统的有向无环图来表示它：

   ![unfolding](../imgs/10/unfolding.png)

2. 如果动态系统有了外部信号驱动：
   $$
   \mathbf{\vec s}^{(t)}=f(\mathbf{\vec s}^{(t-1)},\mathbf{\vec x}^{(t)};\mathbf{\vec \theta})
   $$
   则其展开图如下（左侧为循环图，右侧为展开图。黑色方块，表示单位延时）。为了表明状态 $\mathbf{\vec s}$ 就是网络的隐单元，这里我们用变量 $\mathbf{\vec h}$ 代表状态，重写为：
   $$
   \mathbf{\vec h}^{(t)}=f(\mathbf{\vec h}^{(t-1)},\mathbf{\vec x}^{(t)};\mathbf{\vec \theta})
   $$
   ​

   ![unfolding2](../imgs/10/unfolding2.png)

3. 就像几乎所有函数都可以被认为是前馈网络，基本上任何涉及循环的函数都可以被认为是一个循环神经网络

4. 当训练循环神经网络根据过去预测未来时，网络通常要将 $\mathbf{\vec h}^{(t)}$ 作为过去序列的一个有损的摘要

   - 这个摘要一般是有损的。因为它使用一个固定长度的向量 $\mathbf{\vec h}^{(t)}$ 来映射任意长度的序列 $(\mathbf{\vec x}^{(t)},\mathbf{\vec x}^{(t-1)},\cdots,\mathbf{\vec x}^{(1)})$ 
   - 根据不同的训练准则，摘要可能会有选择地保留过去序列的某些部分

5. 展开图的两个主要优点：

   - 无论输入序列的长度，学得的模型始终具有相同的输入大小（因为在每个时间步上，它从一种状态迁移到另一个状态，其输入都是相同大小的）
   - 可以在每个时间步上相同的转移函数 $f$ ，因此我们需要学的的参数 $\mathbf{\vec \theta}$ 也就在每个时间步上共享

   这就使得学习在所有时间步和所有序列长度上操作的单一模型 $f$ 成为可能；它允许单一模型 $f$ 泛化到没有见过的序列长度；并且学习模型所需的训练样本远少于非参数共享的模型

## 2. 循环神经网络

1.  基于图展开核参数共享的思想，我们可以设计各种循环神经网络。一些重要的设计模式有：

   - 第一类：每个时间步都有输出，并且隐单元之间有循环连接的循环网络：

     > 左图为循环图，右图为计算图。$L$ 为损失函数（衡量每个输出 $\mathbf{\vec o}$ 与标记 $\mathbf{\vec y}$ 的距离 ）

     ![rnn_type1](../imgs/10/rnn_type1.png)

   - 第二类：每个时间步都有输出，只有当前时刻的输出和下个时刻的隐单元之间有循环连接的循环网络。

     - 这种 `RNN`只能表示更小的函数集合。第一类 `RNN`可以选择将其想要的关于过去的任何信息放入隐藏状态表达 $\mathbf{\vec h}$  中，并且通过 $\mathbf{\vec h}$ 传播到未来。而这里的 `RNN` 只有输出 $\mathbf{\vec o}$ 来传播信息到未来。通常 $\mathbf{\vec o}$ 的维度远小于 $\mathbf{\vec h}$ ，并且缺乏过去的重要信息。这使得这种 `RNN` 不那么强大，但是它更容易训练：因为每个时间步可以与其他时间步分离训练，允许训练期间更多的并行化

     ![rnn_type2](../imgs/10/rnn_type2.png)

   - 第三类：隐单元之间存在循环连接，但是读取整个序列之后产生单个输出的循环网络。

     ![rnn_type3](../imgs/10/rnn_type3.png) 

2. 任何图灵可计算的函数都可以通过这样一个有限维的循环网络计算。在这个意义上，第一类`RNN`的循环神经网络时万能的

3. 假设第一类`RNN`使用双曲正切激活函数，输出是离散的（我们将输出 $\mathbf{\vec o}$ 作为每个离散变量可能取值的非标准化对数概率，然后应用 `softmax` 函数处理后，获得标准化概率的输出向量 $\hat{\mathbf {\vec y}}$） 

   `RNN` 从特定的初始状态 $\mathbf{\vec h}^{(0)}$ 开始前向传播。从 $t=1$ 到 $t=\tau$ 的每个时间步，我们更新方程：
   $$
   \mathbf{\vec a}^{(t)}=\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t-1)}+\mathbf U\mathbf{\vec x}^{(t)}\\
   \mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec a}^{(t)})\\
   \mathbf{\vec o}^{(t)}=\mathbf{\vec c}+\mathbf V\mathbf{\vec h}^{(t)}\\
   \hat{\mathbf{\vec y}}^{(t)}=softmax(\mathbf{\vec o}^{(t)})
   $$
   其中：输入到隐藏状态的权重为 $\mathbf U$， 隐藏状态到输出的权重为 $\mathbf V$ ，隐藏状态到隐藏状态的权重为 $\mathbf W$ ；$\mathbf{\vec b},\mathbf{\vec c}$ 为偏置向量

   - 这个循环网络将一个输入序列映射到相同长度的输出序列

   - 给定的 $ \mathbf{\vec x} ,\mathbf{\vec y} $ 的序列对，则总损失为：所有时间步的损失之和。假设 $L^{(t)}$ 为给定 $\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(t)}$ 的条件下，$\mathbf{\vec y}^{(t)}$ 的负对数似然，则：
     $$
     L\left(\{\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(\tau)}\},\{\mathbf{\vec y}^{(1)},\mathbf{\vec y}^{(2)},\cdots,\mathbf{\vec y}^{(\tau)}\}\right)\\
     =\sum_{t=1}^{t=\tau}L^{(t)}\\
     =-\sum_{t=1}^{t=\tau}\log p_{model}\left(\mathbf{\vec y}^{(t)}\mid \{\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(t)}\} \right)
     $$
     其中 $p_{model}\left(\mathbf{\vec y}^{(t)}\mid \{\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(t)}\} \right) $   需要读取模型输出向量 $\hat{\mathbf{\vec y}}^{(t)}$ 中对应于 $\mathbf{\vec y^{(t)}}$ 的项。

     > 因为模型输出向量 $\hat{\mathbf{\vec y}}^{(t)}$ 给出了每个分类的概率，是一个向量。真实的分类 $y^{(t)}$ 属于哪个分类，就提取对应的概率。

   - 这个损失函数的梯度计算是昂贵的。梯度计算涉及执行一次前向传播，一次反向传播。

     - 因为每个时间步只能一前一后的计算，无法并行化，运行时间为 $O(\tau)$ 
     - 前向传播中各个状态必须保存，直到它们反向传播中被再次使用，因此内存代价也是 $O(\tau)$ 

4. 在展开图中代价为 $O(\tau) $ 的反向传播算法称作通过时间反向传播 `back-propagation through time:BPTT` 

### 2.1 Teacher forcing  

1. 对于第二类循环神经网络，由于它缺乏隐藏状态到隐藏状态的循环连接，因此它无法模拟通用图灵机

   - 优点在于：训练可以解耦：各时刻 $t$ 分别计算梯度

2. 第二类循环神经网络模型可以使用 `teacher forcing` 进行训练。该方法在时刻 $t+1$ 接受真实值 $\mathbf{\vec  y}^{(t)}$  作为输入。

   考察只有两个时间步的序列。对数似然函数为
   $$
   \log p(\mathbf{\vec y}^{(1)},\mathbf{\vec y}^{(2)}\mid \mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)})\\
   =\log p(\mathbf{\vec y}^{(2)}\mid \mathbf{\vec y}^{(1)},\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)})+\log p(\mathbf{\vec y}^{(1)}\mid \mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)})
   $$
   对右侧两部分分别取最大化。则可以看到：在时刻 $t=2$ 时，模型被训练为最大化 $\mathbf{\vec y}^{(2)}$ 的条件概率。

   如下图所示为`teacher forcing`过程。左图中，我们将正确的输出  $\mathbf{\vec  y}^{(t)}$  反馈到  $\mathbf{\vec  h}^{(t+1)}$  。一旦模型进行预测时，真实的输出通常是未知的，此时我们用模型的输出  $\mathbf{\vec  o}^{(t)}$  来近似正确的输出  $\mathbf{\vec  y}^{(t)}$ 

   ![teacher_forcing](../imgs/10/teacher_forcing.png) 

3. 采用`teacher forcing`的本质上是：当前隐藏状态与早期的隐藏状态没有连接（虽然有间接连接，但是由于  $\mathbf{\vec  y}^{(t)}$ 已知，因此这种连接被打断）

   - 只要模型的当前时间步的输出与下一个时间步的某个变量存在连接，就可以使用`teacher forcing`来训练
   - 如果模型的隐藏状态依赖于早期时间步的隐藏状态，则需要采用 `BPTT`算法
   - 某些模型训练时，需要同时使用`teacher forcing`和`BPTT`算法

4. `teacher forcing`的缺点是：如果将网络的输出反馈作为输入（这里的输入指的是网络的输入，而不是隐藏状态的输入），此时，训练期间网络看到的输入与测试时看到的输入会有很大的不同

### 2.2 循环神经网络梯度

1. 由反向传播计算得到的梯度，结合任何通用的、基于梯度的技术就可以训练 `RNN`

2. 第一类循环神经网络中，计算图的节点包括参数 $\mathbf U,\mathbf V,\mathbf W,\mathbf{\vec b},\mathbf{\vec c}$，以及以 $t$ 为索引的节点序列 $\mathbf{\vec x}^{(t)},\mathbf{\vec h}^{(t)},\mathbf{\vec o}^{(t)}$ 以及 $L^{(t)}$ 

   > 更新方程为：
   > $$
   > \mathbf{\vec a}^{(t)}=\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t-1)}+\mathbf U\mathbf{\vec x}^{(t)}\\
   > \mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec a}^{(t)})\\
   > \mathbf{\vec o}^{(t)}=\mathbf{\vec c}+\mathbf V\mathbf{\vec h}^{(t)}\\
   > \hat{\mathbf{\vec y}}^{(t)}=softmax(\mathbf{\vec o}^{(t)})
   > $$
   >

   - 根据
     $$
     L\left({\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(\tau)}},{\mathbf{\vec y}^{(1)},\mathbf{\vec y}^{(2)},\cdots,\mathbf{\vec y}^{(\tau)}}\right)\\
     =\sum_{t=1}^{t=\tau}L^{(t)}
     $$
     我们有 $\frac{\partial L}{\partial L^{(t)}}=1$ 

   - 我们假设 $\mathbf{\vec o}^{(t)}$ 作为 `softmax` 函数的参数。可以从 `softmax`获取输出概率的向量 $\hat{\mathbf{\vec y}}^{(t)}$  
     $$
     \hat{\mathbf{\vec y}}^{(t)}=\text{softmax}(\mathbf{\vec o}^{(t)})=\left(\frac{\exp(o^{(t)}_1)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)},\frac{\exp(o^{(t)}_2)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)},\cdots,\frac{\exp(o^{(t)}_n)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)}\right)^{T}
     $$

     > 输出曾其实有多个单元。这里将其展平为一维向量

   - 我们假设损失为：给定了迄今为止的输入后，真实目标 $\mathbf{\vec y}^{(t)}$ 的负对数似然 
     $$
     L^{(t)}=-\log \hat{\mathbf{\vec y}}^{(t)}_l=-\log \frac{\exp(o^{(t)}_l)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)}\\
     =- o^{(t)}_l+\log\sum_{j=1}^{n}\exp(o^{(t)}_j)
     $$
     $l$ 表示 $ \mathbf{\vec y}^{(t)}$ 所属的类别。则：
     $$
     (\nabla_{\mathbf{\vec o}^{(t)}}L)_i=\frac{\partial L}{\partial \mathbf{\vec o}_i^{(t)}}=\frac{\partial L}{\partial L^{(t)}}\frac{\partial L^{(t)}}{\partial \mathbf{\vec o}_i^{(t)}}=\frac{\partial L^{(t)}}{\partial\mathbf{\vec o}_i^{(t)}}\\
     =-\mathbf{\vec 1}_{i,l}+\frac{\exp(\mathbf{\vec o}^{(t)}_i)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)}\\
     =\hat{\mathbf{\vec y}}^{(t)}_i-\mathbf{\vec 1}_{i,l}
     $$
      这里的下标 $i$ 表示输出层的第 $i$ 个单元（它其实就是将所有的输出层展平为一维向量，然后再将结果拆分成一个个单元）。最后一个向量是：第 $i$ 个输出单元的真实类别为 $l$ 处对应为 1；其他地方为0。 $\hat{\mathbf{\vec y}}^{(t)}_i$ 表示输出概率向量对应于输出层第 $i$ 个单元

     > 最后一步是因为 $(\log f(x))^{\prime}=\frac {1}{f(x)}f^{\prime}(x) $，另外 $\frac{\exp(\mathbf{\vec o}^{(t)}_i)}{\sum_{j=1}^{n}\exp(o^{(t)}_j)}=\hat{\mathbf{\vec y}}^{(t)}_i$

   - 从后向前计算隐单元：
     $$
     \nabla_{\mathbf{\vec h}^{(t)}}L=\left(\frac{\partial\mathbf{\vec h}^{(t+1)}}{\partial\mathbf{\vec h}^{(t)} }\right)^{T}(\nabla_{\mathbf{\vec h}^{(t+1)}}L)+\left(\frac{\mathbf{\vec o}^{(t)}}{\partial\mathbf{\vec h}^{(t)} }\right)^{T}(\nabla_{\mathbf{\vec o}^{(t)}}L)\\
     =\mathbf W^{T}(\nabla_{\mathbf{\vec h}^{(t+1)}}L)\text{diag}\left(1-(\mathbf{\vec h}^{(t+1)})^{2}\right)+\mathbf V^{T}(\nabla_{\mathbf{\vec o}^{(t)}}L)
     $$

     > 因为有
     > $$
     > \mathbf{\vec h}^{(t+1)}=\tanh(\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t)}+\mathbf U\mathbf{\vec x}^{(t+1)})
     > $$
     > 因此
     > $$
     > \left(\frac{\partial\mathbf{\vec h}^{(t+1)}}{\partial\mathbf{\vec h}^{(t)} }\right)^{T}=\mathbf W^{T}\text{diag}\left(1-(\mathbf{\vec h}^{(t+1)})^{2}\right)
     > $$
     > ​

     其中 $\text{diag}\left(1-(\mathbf{\vec h}^{(t+1)})^{2}\right)$  表示包含元素 $1-(h_i^{(t+1)})^{2}$ 的对角矩阵（它是时刻 $t+1$ 时，隐单元 $i$ 对应的 $\tanh$ 的导数）

   - 上式计算隐单元的梯度时，注意它计算的是 $t=\tau-1 ,\cdots,1$ 的情况。 当 $t=\tau$ 时，位于序列末尾。此时并没有 $\mathbf{\vec h}^{(\tau+1)}$，因此有：
     $$
     \nabla_{\mathbf{\vec h}^{(t)}}L=\mathbf V^{T}(\nabla_{\mathbf{\vec o}^{(t)}}L)
     $$

3. 一旦获得了隐单元及输出单元的梯度，则我们就可以获取参数节点的梯度。

   -  由于参数在多个时间步共享，因此在参数节点的微积分操作时必须谨慎对待
   -  微积分中的算子 $\nabla _{\mathbf W}f$ ，在计算 $\mathbf W$ 对于 $f$ 的贡献时，将计算图的所有边都考虑进去了。但是事实上，有一条边是 $t$ 时间步的 $\mathbf W$；还有一条边是 $t+1$ 时间步的 $\mathbf W$ 
   -  为了消除歧义，我们使用虚拟变量 $\mathbf W^{(t)}$ 作为 $\mathbf W$ 的副本，用 $\nabla _{\mathbf W^{(t)}} f$ 表示权重 $\mathbf W$ 在时间步 $t$ 对于梯度的贡献

4. 剩下的参数梯度公式如下：
   $$
   \nabla _{\mathbf{\vec c}}L=\sum_{t=1}^{t=\tau}\left(\frac{\partial \mathbf{\vec o}^{(t)}}{\partial \mathbf{\vec c}}\right)^{T}\nabla_{\mathbf{\vec o}^{(t)}}L=\sum_{t=1}^{t=\tau}\nabla_{\mathbf{\vec o}^{(t)}}L
   $$

   > 这是因为 $\mathbf{\vec o}^{(t)}=\mathbf{\vec c}+\mathbf V\mathbf{\vec h}^{(t)}\rightarrow \frac{\partial \mathbf{\vec o}^{(t)}}{\partial \mathbf{\vec c}}=\mathbf I​$

   $$
   \nabla _{\mathbf{\vec b}}L=\sum_{t=1}^{t=\tau}\left(\frac{\partial \mathbf{\vec h}^{(t)}}{\partial \mathbf{\vec b}}\right)^{T}\nabla_{\mathbf{\vec h}^{(t)}}L=\sum_{t=1}^{t=\tau}\text{diag}\left(1-(\mathbf{\vec h}^{(t)})^{2}\right)\nabla_{\mathbf{\vec h}^{(t)}}L
   $$

   > 这是因为 $\mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t-1)}+\mathbf U\mathbf{\vec x}^{(t)}) \rightarrow \frac{\partial \mathbf{\vec h}^{(t)}}{\partial \mathbf{\vec b}}=\text{diag}\left(1-(\mathbf{\vec h}^{(t)})^{2}\right)$

   $$
   \nabla_{\mathbf V}L=\sum_{t=1}^{t=\tau}\sum_{i}\left(\frac{\partial L}{\partial \mathbf{\vec o}_i^{(t)}}\right)^{T}\nabla_{\mathbf V}\mathbf{\vec o}_i^{(t)}=\sum_{t=1}^{t=\tau}(\nabla_{\mathbf{\vec o}^{(t)}}L)\mathbf{\vec h}^{(t)T}
   $$

   > 这是因为 $\mathbf{\vec o}^{(t)}=\mathbf{\vec c}+\mathbf V\mathbf{\vec h}^{(t)} \rightarrow \nabla_{\mathbf V}\mathbf{\vec o}_i^{(t)}=\mathbf{\vec h}^{(t)T}_i$  

   $$
   \nabla_{\mathbf W}L=\sum_{t=1}^{t=\tau}\sum_{i}\left(\frac{\partial L}{\partial \mathbf{\vec h}_i^{(t)}}\right)^{T}\nabla_{\mathbf W^{(t)}}\mathbf{\vec h}_i^{(t)}\\
   =\sum_{t=1}^{t=\tau}\text{diag}\left(1-(\mathbf{\vec h}^{(t)})^{2}\right)(\nabla_{\mathbf{\vec h}^{(t)}}L)\mathbf{\vec h}^{(t-1)T}
   $$

   > 这是因为 $\mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t-1)}+\mathbf U\mathbf{\vec x}^{(t)}) \rightarrow\nabla_{\mathbf W^{(t)}}\mathbf{\vec h}_i^{(t)}=\text{diag}\left(1-(\mathbf{\vec h}^{(t)}_i)^{2}\right)\mathbf{\vec h}^{(t-1)T}_i$ 

   $$
   \nabla_{\mathbf U}L=\sum_{t=1}^{t=\tau}\sum_{i}\left(\frac{\partial L}{\partial \mathbf{\vec h}_i^{(t)}}\right)^{T}\nabla_{\mathbf U^{(t)}}\mathbf{\vec h}_i^{(t)}\\
   =\sum_{t=1}^{t=\tau}\text{diag}\left(1-(\mathbf{\vec h}^{(t)})^{2}\right)(\nabla_{\mathbf{\vec h}^{(t)}}L)\mathbf{\vec x}^{(t)T}
   $$

   > 这是因为 $\mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec b}+\mathbf W\mathbf{\vec h}^{(t-1)}+\mathbf U\mathbf{\vec x}^{(t)}) \rightarrow\nabla_{\mathbf U^{(t)}}\mathbf{\vec h}_i^{(t)}=\text{diag}\left(1-(\mathbf{\vec h}^{(t)}_i)^{2}\right)\mathbf{\vec x}^{(t)T}_i$ 

5. 因为任何参数都不是训练数据 $\mathbf{\vec x}^{(t)}$ 的父节点，因此我们不需要计算 $\nabla _{\mathbf{\vec x}^{(t)}} L$ 

### 2.3 作为有向图模型的循环网络

1. 目前我们将循环神经网络的损失 $L^{(t)}$ 定义为训练目标 $\mathbf{\vec y}^{(t)}$ 和输出 $\mathbf{\vec o}^{(t)}$ 之间的交叉熵

   - 原则上，你可以使用任何损失函数。但是必须根据具体任务来选择一个合适的损失函数

2. 对于 `RNN`，我们有两个选择：

   - 根据之前的输入，估计下一个序列元素 $ \mathbf{\vec y}^{(t)}$ 的条件分布，即最大化对数似然函数：
     $$
     \log p(\mathbf{\vec y}^{(t)}\mid \mathbf{\vec x}^{(1)},\cdots,\mathbf{\vec x}^{(t)})
     $$

   - 如果模型包含了来自于一个时间步到下一个时间步的连接（比如第二种类型的循环神经网络），则最大化对数似然函数：

     $$
     \log p(\mathbf{\vec y}^{(t)}\mid \mathbf{\vec      x}^{(1)},\cdots,\mathbf{\vec x}^{(t)},\mathbf{\vec      y}^{(1)},\cdots,\mathbf{\vec y}^{(t-1)})
     $$

     > 当我们不把过去的 $\mathbf{\vec y}$ 反馈给下一步时，有向图模型就不包含任何从过去的 $\mathbf{\vec y}^{(i)}$ 到当前的  $\mathbf{\vec y}^{(t)}$ 的边

3. 循环神经网络的有向图模型：不包含任何输入。

   - 当模型包含了来自于一个时间步到下一个时间步的连接时，有向图非常复杂。根据有向图直接参数化的做法非常低效

     ![rnn_dag](../imgs/10/rnn_dag.png)

   - 相反：`RNN `引入了状态变量。它有助于我们获取非常高效的参数化：序列中的每个时间步使用相同的结构，并且共享相同的参数
     - 引入状态变量 $\mathbf{\vec h}^{(t)}$ 节点，可以用作过去和未来之间的变量，从而解耦它们。遥远的过去的变量 $\mathbf{\vec y}^{(i)}$ 通过影响 $\mathbf{\vec h}$ 来影响变量 $\mathbf{\vec y}^{(t)}$ 

   ​    ![rnn_dag2](../imgs/10/rnn_dag2.png)

4. 当模型包含单个向量 $\mathbf{\vec x}$ 作为输入时，有三种方式：

   - 在每个时刻作为一个额外的输入。如图注任务：单个图像作为模型输入，生成描述图像的单词序列

     此时引入了新的权重矩阵 $\mathbf R$ 。 乘积 $\mathbf{\vec x}^{T}\mathbf R$ 在每个时间步作为隐单元的一个额外输入

     > 输出序列的每个元素 $y^{(t)}$ 同时用作输入（对于当前时间步），和训练期间的目标（对于前一个时间步）。因为它需要根据当前的序列和图像，预测下一个时刻的单词

     ![rnn_x](../imgs/10/rnn_x.png)

   - 作为初始状态 $\mathbf{\vec h}^{(0)}$

   - 结合上述两种方式

5. `RNN`也可以接收向量序列 $\mathbf{\vec x^{(t)}}$ 作为输入。此时`RNN`对应条件分布为 $P( \mathbf{\vec y}^{(1)},\cdots,\mathbf{\vec y}^{(\tau)}\mid \mathbf{\vec x}^{(1)},\cdots,\mathbf{\vec x}^{(\tau)})$ 

   - 第一类循环神经网络假设条件独立，此时： 
     $$
     P( \mathbf{\vec y}^{(1)},\cdots,\mathbf{\vec y}^{(\tau)}\mid \mathbf{\vec x}^{(1)},\cdots,\mathbf{\vec x}^{(\tau)})=\prod_tP( \mathbf{\vec y}^{(t)}\mid \mathbf{\vec x}^{(1)},\cdots,\mathbf{\vec x}^{(t)})
     $$

   - 为了去掉条件独立的假设，我们可以在时刻 $t$ 的输出到时刻 $t+1$ 的隐单元添加连接。此时该模型代表了关于 $\mathbf{\vec y}$  序列的任意概率分布

     - 但是这样的序列有个限制：要求输入序列和输出序列的长度必须相同

       ![rnn_more](../imgs/10/rnn_more.png) 

6. 循环网络中使用参数共享的前提是：相同参数可以用于不同的时间步。

   - 给定时刻 $t$ 的变量，时刻 $t+1 $ 变量的条件概率分布是平稳的。这意味着：前一个时间步和后一个时间步之间的关系并不依赖于 $t$ 

7. 对于不包含输入，或者只有单个输入向量的`RNN`模型，必须有某种办法来确定输出序列的长度（不能无限长）

   > 如果不包含输入，则训练集中的样本只有输出数据。这种 `RNN` 可以用于自动生成文章、句子等

   - 方法一：当输出时单词时，我们可以添加一个特殊的停止符。当输出该停止符时，序列终止。
     - 在训练集中，需要对每个输出序列末尾添加这个停止符
   - 方法二：在模型中引入一个额外的伯努利输出单元，用于表示：每个时间步是继续生成输出序列，还是停止生成。
     - 这种办法更普遍，适用于任何`RNN`
     - 该伯努利输出单元通常使用`sigmoid`单元，被训练为最大化正确预测到序列结束的对数似然
   - 方法三：添加一个额外的输出单元，它预测输出序列的长度 $\tau$ 本身
     - 这种方法需要在每个时间步的循环更新中增加一个额外输入，通知循环：是否已经到达输出序列的末尾
     - 如果没有这个额外的输入，则`RNN`很可能在生成完整的句子之前突然结束序列
     - 其原理是基于 $P(\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(\tau)})=P(\tau)P(\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(\tau)}\mid\tau)$  

## 3. 双向 RNN

1. 目前介绍的循环神经网络有一个因果结构：时刻 $t$ 的状态只能从过去的序列 $\mathbf{\vec x}^{(t1)},\mathbf{\vec x}^{(t2)},\cdots,\mathbf{\vec x}^{(t-1)} $ ，以及当前的输入 $\mathbf{\vec x}^{(t)}$ 来获取

   - 实际应用中，输出 $\mathbf{\vec y}^{(t)}$ 可能依赖于整个输入序列。如：语音识别中，当前语音对应的单词不仅取决于前面的单词，也可能取决于后面的几个单词。因为词与词之间存在语义依赖

2. 双向`RNN`就是应对这种双向依赖问题而发明的。

   - 它在需要双向信息的应用中非常成功，如手写识别、语音识别等。

   - 下图是一个典型的双向 `RNN`

     - $\mathbf{\vec h}^{(t)}$ 代表通过时间向前移动的子 `RNN`的状态，向右传播信息

     - $\mathbf{\vec g}^{(t)}$ 代表通过时间向后移动的子 `RNN` 的状态，向左传播信息

     - 输出单元 $\mathbf{\vec o}^{(t)}$ 能够表示同时依赖于过去和未来、以及时刻 $t$ 的输入 $\mathbf{\vec x}^{(t)}$ 的概要

       > 如果使用前馈网络、卷积网络，则我们需要指定 $t$ 周围固定大小的窗口

     ![bi_RNN](../imgs/10/bi_RNN.png)

3. 如果输入是 2 维的，比如图像，则双向 `RNN` 可以扩展到4个方向：上、下、左、右

   - 每个子 `RNN` 负责一个时间移动方向
   - 输出单元 $\mathbf{\vec o}^{(t)}$ 能够表示同时依赖于四个方向、以及时刻 $t$ 的输入 $\mathbf{\vec x}^{(t)}$ 的概要
   - 此时`RNN` 可以捕捉到大多数局部信息，但是还可以捕捉到依赖于远处的信息
   - 相比较卷积网络，应用于图像的`RNN`计算成本通常更高

## 4. 基于编码-解码的序列到序列架构

1. 如前所述：

   - 我们可以通过`RNN`将输入序列映射成固定大小的向量

     ![rnn_type3](../imgs/10/RNN_type3.png)

   - 我们也可以通过`RNN`将固定大小的向量映射成一个序列

     ![rnn_x](../imgs/10/rnn_x.png)

   - 我们也可以通过`RNN`将一个输入序列映射到一个等长的输出序列

     ![rnn_type3](../imgs/10/RNN_type1.png)

     ![rnn_type3](../imgs/10/rnn_more.png)

     ​

2. 在实际应用中，我们需要将输入序列映射到不一定等长的输出序列。如语音识别、机器翻译、问答等任务

3. 我们将`RNN`的输入称作上下文。令 $C$  为该上下文的一个表达`representation`

   - $C$ 需要概括输入序列 $\mathbf X=(\mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(nx)})$ 
   - $C$ 可能是一个向量，或者一个向量序列

4. 编码-解码架构：

   - 编码器（也叫作读取器，或者输入`RNN`）处理输入序列。编码器的最后一个状态 $\mathbf{\vec h}_{nx}$ 通常最为输入的表达 $C$，并且作为解码器的输入 
   - 解码器（也叫作写入器，或者输出`RNN`）以固定长度的向量（就是 $C$ ）为条件产生输出序列 $\mathbf Y=(\mathbf{\vec y}^{(1)},\mathbf{\vec y}^{(2)},\cdots,\mathbf{\vec y}^{(ny)})$ 。

5. 这种架构创新之处在于：输入序列长度 $nx$ 和输出序列长度 $ny$ 可以不同。而前面的架构要求 $nx=ny=\tau$ 。

6. 这两个`RNN`并不是单独训练，而是共同训练以最大化 $\log P(\mathbf{\vec y}^{(1)},\mathbf{\vec y}^{(2)},\cdots,\mathbf{\vec y}^{(ny)}\mid \mathbf{\vec x}^{(1)},\mathbf{\vec x}^{(2)},\cdots,\mathbf{\vec x}^{(nx)})$ 

7. 如果上下文 $C$ 是一个向量，则解码器就是向量到序列`RNN`。向量到序列`RNN` 至少有两种接收输入的方法：

   - 输入作为`RNN`的初始状态
   - 输入连接到每个时间步的隐单元

   这两种方式也可以结合

8. 这里对于编码器与解码器隐藏状态是否具有相同大小并没有限制。它们是相互独立设置的

9. 该架构的一个明显缺点：编码器`RNN`输出的上下文 $C$ 的维度太小，难以适当的概括一个长序列

   - 可以让 $C$ 也成为一个可变长度的序列
   - 引入将序列 $C$ 的元素和输出序列元素相关联的`attention`机制

## 5. 深度循环网络

1. 大多数`RNN`中的计算都可以分解为三块参数以及相关的变换

   - 从输入到隐藏状态的变换
   - 从前一个隐藏状态到下一个隐藏状态的变换
   - 从隐藏状态到输出的变换

   这三块都对应一个浅层变换。所谓浅层变换：多层感知机内单个层来表示的变换称作浅层变换。通常浅层变换是由一个仿射变换加一个固定的非线性函数（激活函数）组成的变换。

2. 我们可以将`RNN`的状态分为多层。层次结构中较低层的作用是：将输入转化为对高层隐藏状态更合适的表达。下图中，将隐藏状态拆分成两层

   ![more_hiden](../imgs/10/more_hiden.png)

3. 也可以在这三块中各自使用一个独立的`MLP`（可能是深度的）

   ![deep_RNN](../imgs/10/deep_rnn.png)

   - 增加深度可能会因为优化困难而损坏学习效果

   - 额外深度将导致从时间步 $t$  到时间步 $t+1$ 的最短路径变得更长。这时可以通过在隐藏状态到隐藏状态的路径中引入跳跃连接即可缓解该问题

     ![deep_RNN2](../imgs/10/deep_rnn2.png)

## 6. 递归神经网络

1. 递归神经网络时循环网络的另一个扩展：它被构造为深的树状而不是链状

   - 它是一种不同类型的计算图
   - 这类网络的潜在用途：学习推论
   - 递归神经网络目前成功用于输入是数据结构的神经网络，如：自然语言处理，计算机视觉

   ![tree_rnn](../imgs/10/tree_rnn.png)

2. 递归神经网络的显著优势：对于长度为 $\tau$ 的序列，深度可以急剧的从 $\tau$ 降低到 $O(\log\tau)$。 这有助于解决长期依赖

3. 一个悬而未决的难题是：如何以最佳的方式构造树

   - 一种选择是：不依赖于数据的树。如：平衡二叉树
   - 另一种选择是：可以借鉴某些外部方法。如处理自然语言时，用于递归网络的树可以固定为句子语法分析树的结构

## 7. 长期依赖

1. 长期依赖产生的根本问题是：经过许多阶段传播之后，梯度趋向于消失（大部分情况）或者爆炸（少数情况，但是一旦发生就对优化过程影响巨大）

2. 循环神经网络设计许多相同函数的多次组合，每个时间步一次。这些组合可以导致极端的非线性行为。

3. 给定一个非常简单的循环联系（没有非线性，没有偏置）
   $$
   \mathbf{\vec h}^{(t)}=\mathbf W^{T}\mathbf{\vec h}^{(t-1)}
   $$
   则有： $\mathbf{\vec h}^{(t)}=(\mathbf W^{t})^{T}\mathbf{\vec h}^{(0)}$ 

   当 $\mathbf W$ 可以正交分解时： $\mathbf W=\mathbf Q\mathbf \Lambda \mathbf Q^{T}$ 。其中 $\mathbf Q$ 为正交矩阵，$\mathbf \Lambda$ 为特征值组成的三角阵。则：
   $$
   \mathbf{\vec h}^{(t)}=\mathbf Q^{T}\mathbf\Lambda^{t}\mathbf Q\mathbf{\vec h}^{(0)}
   $$

   - 对于特征值的幅度不到 1 的特征值对应的 $\mathbf{\vec h}^{(0)}$ 的部分将衰减到 0
   - 对于特征值的幅度大于 1 的特征值对应的 $\mathbf{\vec h}^{(1)}$ 的部分将衰减到 0

4. 对于非循环神经网络，情况稍微好一些。

   - 对于标量权重 $w$，假设每个时刻使用不同的权重 $w^{(t)}$ （初值为 1），而且 $w^{(t)}$ 是随机生成、各自独立、均值为 0、方差为 $v$ 。则 $\prod_t w^{(t)}$ 的方差为 $O(v^{n})$
   - 非常深的前馈网络通过精心设计的比例可以避免梯度消失核爆炸问题

5. 学习长期依赖的问题是深度学习中的一个主要挑战

## 8. 回声状态网络

1. 从 $\mathbf{\vec h}^{(t-1)}$ 到 $\mathbf{\vec h}^{(t)}$ 的循环映射权重，以及从  $\mathbf{\vec x}^{(t)}$ 到 $\mathbf{\vec h}^{(t)}$ 的输入映射权重是循环网络中最难学习的参数

   一种解决方案是：设定循环隐藏单元，它能够很好地捕捉过去输入历史，并且只学习输出权重。这就是回声状态网络 `echo state netword:ESN`

   > 流体状态机`liquid state machine`也分别独立地提出了这种想法

2. 回声状态网络和流体状态机都被称作储层计算`reservoir computing`。因为隐藏单元形成了可能捕获输入历史不同方面的临时特征池

3. 储层计算循环网络类似于支持向量机的核技巧：

   - 先将任意长度的序列（到时刻 $t$ 的输入历史） 映射为一个长度固定的向量 (循环状态 $\mathbf {\vec h}^{(t)}$) 
   - 然后对 $\mathbf{\vec h}^{(t)}$ 施加一个线性预测算子（通常是一个线性回归） 以解决感兴趣的问题

4. 储层计算循环网络的目标函数很容易设计为输出权重的凸函数。因此很容易用简单的学习算法解决。

   - 如果隐单元到输出单元是线性回归，则损失函数就是均方误差

5. 储层计算循环网络的核心问题：如何将任意长度的序列（到时刻 $t$ 的输入历史） 映射为一个长度固定的向量 (循环状态 $\mathbf {\vec h}^{(t)}$) ？

   答案是：将网络视作动态系统，并且让动态系统接近稳定边缘。

6. 考虑反向传播中，雅克比矩阵 $\mathbf J^{(t)}=\frac{\partial\mathbf{\vec s}^{(t)}}{\partial \mathbf{\vec s}^{(t-1)}}$。 定义其谱半径为特征值的最大绝对值。

7. 考虑反向传播中，雅克比矩阵 $\mathbf J^{(t)}=\frac{\partial\mathbf{\vec s}^{(t)}}{\partial \mathbf{\vec s}^{(t-1)}}$。 定义其谱半径为特征值的最大绝对值。

   - 假设网络是纯线性的，此时 $\mathbf J^{(t)}=\mathbf J$ 。假设 $\mathbf J$ 特征值 $\lambda $ 对应的单位特征向量为 $\mathbf{\vec v}$ 
   - 设刚开始的梯度为 $\mathbf{\vec g}$，经过 $n$ 步反向传播之后，梯度为 $\mathbf J^{n}\mathbf{\vec g}$
   - 假设开始的梯度有一个扰动，为 $\mathbf{\vec g}+\delta \mathbf{\vec v}$，经过 $n$ 步反向传播之后，梯度为 $\mathbf J^{n}\mathbf{\vec g}+\delta J^{n}\mathbf{\vec v}$ 。则这个扰动在 $n$ 步之后放大了 $\delta |\lambda|^{n}$ 倍。
     - 当 |$\lambda|>1$ 时，偏差 $\delta|\lambda|^{n}$ 就会指数增长
     - 当  |$\lambda|<1$ 时，偏差 $\delta|\lambda|^{n}$ 就会指数衰减

   这个例子对应于没有非线性的循环网络。当非线性存在时：非线性的导数将在许多个时间步之后接近 0，这有助于防止因过大的谱半径而导致的偏差爆炸

8. 在正向传播的情况下， $\mathbf W$ 告诉我们信息如何前向传播；在反向传播的情况下， $\mathbf J$ 告诉我们梯度如何后向传播

   - 当循环神经网络时线性的情况时，二者相等
   - 当引入压缩非线性时（如 `sigmoid/tanh`），此时可以使得 $\mathbf h^{(t)}$ 有界（即前向传播有界），但是梯度仍然可能无界（即反向传播无界）
   - 在神经网络中，反向传播更重要。因为我们需要使用梯度下降法求解参数！

9. 回声状态网络的策略是：简单地固定权重，使得它具有一定的谱半径（比如 3）

   - 在信息前向传播过程中，由于饱和非线性单元的稳定作用， $\mathbf h^{(t)}$ 不会爆炸
   - 在信息反向传播过程中，非线性的导数将在许多个时间步之后接近 0，梯度也不会发生爆炸


## 9. 渗漏单元和其他多时间尺度的策略

1. 处理长期依赖的一个策略是：
   - 设计工作在多个时间尺度的模型
   - 在细粒度的时间尺度上处理近期信息；在粗粒度时间尺度上处理遥远过去的信息

### 9.1 跳跃连接

1. 增加从遥远过去的变量到当前变量的直接连接是得到粗时间尺度的一种方法
   - 普通的循环网络中，循环从时刻 $t$ 单元连接到了时刻 $t+1$ 单元
   - 跳跃连接会增加一条从时刻 $t$ 到时刻 $t+d$  单元的连接
2. 引入了 $d$ 延时的循环连接可以减轻梯度消失的问题
   - 现在梯度指数降低的速度与 $\frac{\tau}{d}$ 相关，而不是与 $\tau$ 相关。这允许算法捕捉到更长的依赖性
   - 这种做法无法缓解梯度的指数爆炸的问题

### 9.3 渗漏单元

1. 缓解梯度爆炸和梯度消失的一个方案是：尽可能的使得梯度接近1.这是通过线性自连接单元来实现

2. 一个线性自连接的例子： 
   $$
   \mu^{(t)}=\alpha \mu^{(t-1)}+(1-\alpha)v^{(t)}
   $$
   当 $\alpha$ 接近1 时， $\mu^{(t)}$ 能记住过去很长一段时间的信息；当 $\alpha$ 接近 0 时，关于过去的信息被迅速丢弃

3. 线性自连接的隐单元可以模拟 $\mu^{(t)}$  的行为。这种隐单元称作渗漏单元

4. 渗漏单元与跳跃连接的区别：

   - $d$ 时间步的跳跃连接可以确保单元总能够被先前的 $d$ 个时间步的值所影响
   - 渗漏单元可以通过调整 $\alpha$ 值更灵活的确保单元可以访问到过去的值

5. 渗漏单元和跳跃连接的 $\alpha,d$ 参数有两种设置方式：

   - 一种方式是手动设置为常数。如初始化时从某些分布采样它们的值
   - 另一种方式是让他们成为自由变量，从训练中学习出来

### 9.3 删除连接

1. 删除连接与跳跃连接不同：删除连接是主动删除长度为 1 的连接，并且用更长的连接替换它们
   - 删除连接会减少计算图中的连接，而不是增加连接
   - 删除连接强迫单元在长时间尺度上运作。而跳跃连接可以选择在长时间尺度上运作，也可以在短时间尺度上运作。
2. 强制一组循环单元在不同时间尺度上运作有不同的方式：
   - 一种选择是：让循环单元变成渗漏单元，但是不同的单元组有不同的、固定的 $\alpha$ 值
   - 另一种选择是：使得显式、离散的更新发生在不同的时间，且不同的单元组有不同的更新频率

## 10. LSTM 和其他门控RNN

1. 目前实际应用中最有效的序列模型称作门控RNN，包括基于 LSTM`long short-term memory` 和 基于门控循环单元`gated recurrent unit`网络
2. 门控RNN 的思路和渗漏单元一样：生成通过时间的路径，其中导数既不消失也不爆炸
   - 渗漏单元选择手动选择常量的连接权重（如跳跃连接）或参数化的连接权重（如渗漏单元）来达到这个目的
   - 门控 RNN 将其推广为在每个时间步都可能改变的连接权重
3. 渗漏单元允许网络在较长持续时间内积累信息
   - 这个方法有个缺点：有时候我们希望一旦这个信息被使用（即被消费掉了），那么这个信息就要被遗忘（丢掉它，使得它不再继续传递）
   - 门控RNN学会决定何时清除信息，而不需要手动决定

### 10.1 LSTM

1. LSTM引入自循环以产生梯度长时间持续流动的路径

   - 其中一个关键扩展是：自循环的权重视上下文而定，而不是固定的
   - 通过门控这个自循环的权重（由另一个隐单元控制），累计的时间尺度可以动态改变。即使是固定参数的 LSTM，累计的时间尺度也会因为输入序列而改变，因为时间常数时模型本身的输出
   - LSTM 在手写识别、语音识别、机器翻译、为图像生成标题等领域获得重大成功

2. LSTM循环网络除了外部的 RNN 循环之外，还有内部的 LSTM细胞循环（自环）。LSTM的细胞代替了普通 RNN 的隐单元

   - 细胞最重要的组成部分是细胞状态 $\mathbf{\vec s}^{(t)}$ ，它不是 $\mathbf{\vec h}^{(t)}$ 

   - 与渗漏单元类似，LSTM 细胞也有线性自环。但是这里自环的权重不再是常数 $\alpha$ ，而是由遗忘门 $f_i^{(t)}$  控制的
     $$
     f_i^{(t)}=\sigma(b_i^{f}+\sum_jU_{i,j}^{f}x_j^{(t)}+\sum_jW_{i,j}^{f}h_j^{(t-1)})
     $$
     写成向量形式为：($\sigma$ 为逐元素的函数)
     $$
     \mathbf{\vec f}^{(t)}=\sigma(\mathbf{\vec b}^{f}+\mathbf U^{f}\mathbf{\vec x}^{(t)}+\mathbf W^{f}\mathbf{\vec h}^{(t-1)})
     $$
     遗忘门通过 `sigmoid`单元将权重设置为 0,1 之间的值。它代替了手动设置渗漏单元的 $\alpha $。其中 $\mathbf{\vec b}^{f},\mathbf U^{f},\mathbf W^{f}$ 分别为遗忘门的偏置、遗忘门的输入权重、遗忘门的循环权重

   - 细胞状态更新为：
     $$
     s_i^{(t)}=f_i^{(t)}s_i^{(t-1)}+g_i^{(t)}\sigma\left(b_i+\sum_jU_{i,j}x_j^{(t)}+\sum_jW_{i,j}h_j^{(t-1)}\right)
     $$
     写成向量的形式为： ($\sigma$ 为逐元素的函数， $\odot$ 为逐元素的向量乘积)
     $$
     \mathbf{\vec s}^{(t)}=\mathbf{\vec f}^{(t)}\odot\mathbf{\vec s}^{(t-1)}+\mathbf{\vec g}^{(t)}\odot \sigma(\mathbf{\vec b}+\mathbf U\mathbf {\vec x}^{(t)}+\mathbf W\mathbf{\vec h}^{(t-1)})
     $$
     其中 $\mathbf {\vec b},\mathbf U,\mathbf W$ 分别为细胞的偏置、细胞的输入权重、细胞的循环权重。 $\mathbf{\vec g}^{(t)}$ 类似于遗忘门，它称作：细胞的外部输入门单元

   - 细胞的外部输入门的更新方程：
     $$
     g_i^{(t)}=\sigma(b_i^{g}+\sum_jU_{i,j}^{g}x_j^{(t)}+\sum_jW_{i,j}^{g}h_j^{(t-1)})
     $$
     写成向量的形式为：($\sigma$ 为逐元素的函数)
     $$
     \mathbf{\vec g}^{(t)}=\sigma(\mathbf{\vec b}^{g}+\mathbf U^{g}\mathbf{\vec x}^{(t)}+\mathbf W^{g}\mathbf{\vec h}^{(t-1)})
     $$
     其中 $\mathbf{\vec b}^{g},\mathbf U^{g},\mathbf W^{g}$ 分别为输入门的偏置、输入门的输入权重、输入门的循环权重

   - 细胞的输出就是 $\mathbf{\vec h}^{(t)}$。它由细胞的输出门控制：
     $$
     h_i^{(t)}=\tanh(s_i^{(t)})q_i^{(t)}
     $$

     $$
     q_i^{(t)}=\sigma(b_i^{o}+\sum_jU_{i,j}^{o}x_j^{(t)}+\sum_jW_{i,j}^{o}h_j^{(t-1)})
     $$

     写成向量的形式： ($\sigma,\tanh$ 为逐元素的函数， $\odot$ 为逐元素的向量乘积)
     $$
     \mathbf{\vec h}^{(t)}=\tanh(\mathbf{\vec s}^{(t)})\odot\mathbf{\vec q}^{(t)}
     $$

     $$
     \mathbf{\vec q}^{(t)}=\sigma(\mathbf{\vec b}^{o}+\mathbf U^{o}\mathbf{\vec x}^{(t)}+\mathbf W^{o}\mathbf{\vec h}^{(t-1)})
     $$

     其中 $\mathbf{\vec b}^{o},\mathbf U^{o},\mathbf W^{o}$ 分别为输出门的偏置、输出门的输入权重、输出门的循环权重

   - 一旦得到了细胞的输出 $\mathbf{\vec h}^{(t)}$，则获取整个循环单元的输出 $\mathbf{\vec o}$ 就和普通的 RNN相同。另外下一步循环单元的连接也是通过 $\mathbf{\vec h}^{(t)}$ 来连接的，与普通 RNN相同

3. 你可以可以选择使用细胞状态 $\mathbf{\vec s}^{(t)}$ 作为门的额外输入，如下图所示

   - 此时 $\mathbf{\vec f}^{(t)},\mathbf{\vec g}^{(t)},\mathbf{\vec q}^{(t)}$ 就多了额外的权重参数（对应于 $\mathbf{\vec s}^{(t-1)}$ 的权重）

   ![LSTM](../imgs/10/LSTM.png) 

   ​

### 10.2 其他门控 RNN

1. 门控循环单元`GRU` 与`LSTM`的主要区别是：门控循环单元的单个门控单元同时控制遗忘因子、更新状态单元

2. `GRU`的细胞输出更新公式：
   $$
   h_i^{(t)}=u_i^{(t-1)}h_i^{(t-1)}+(1-u_i^{(t-1)})\sigma\left(b_i+\sum_jU_{i,j}x_j^{(t)}+\sum_jW_{i,j}r_j^{(t-1)}h_j^{(t-1)}\right)
   $$
   写成向量的形式：其中 $\odot$ 为逐元素的向量乘积； $\sigma$ 为逐元素的函数
   $$
   \mathbf{\vec h}^{(t)}=\mathbf{\vec u}^{(t-1)}\odot\mathbf{\vec h}^{(t-1)}+(1-\mathbf{\vec u}^{(t-1)})\odot\sigma(\mathbf{\vec b}+\mathbf U\mathbf{\vec x}^{(t)}+\mathbf W\mathbf{\vec r}^{(t-1)}\odot \mathbf{\vec h}^{(t-1)})
   $$

   - 其中 $u$ 为更新门，其公式为：
     $$
     u_i^{(t)}=\sigma\left(b_i^{u}+\sum_jU^{u}_{i,j}x_j^{(t)}+\sum_jW_{i,j}^uh_j^{(t)}\right)
     $$
     写成向量的形式： $\sigma$ 为逐元素的函数
     $$
     \mathbf{\vec u}^{(t)}=\sigma(\mathbf{\vec b}^{u}+\mathbf U^{u}\mathbf{\vec x}^{(t)}+\mathbf W^{u}\mathbf{\vec h}^{(t)})
     $$

   - 其中 $r$ 为复位门，其公式为：
     $$
     r_i^{(t)}=\sigma\left(b_i^{r}+\sum_jU^{r}_{i,j}x_j^{(t)}+\sum_jW_{i,j}^rh_j^{(t)}\right)
     $$
     写成向量的形式： $\sigma$ 为逐元素的函数
     $$
     \mathbf{\vec r}^{(t)}=\sigma(\mathbf{\vec b}^{r}+\mathbf U^{r}\mathbf{\vec x}^{(t)}+\mathbf W^{r}\mathbf{\vec h}^{(t)})
     $$

3. 复位门和更新门能够独立的“忽略”状态向量的一部分

   - 更新门可以选择是采用旧的 $\mathbf{\vec h}$ 值，还是完全用新的值 ($\sigma(.)$ 给出的） 来给出新的 $\mathbf{\vec h}$ ；或者是两者之间
   - 复位门控制了当前状态中哪些用于计算下一个状态

4. 围绕这一主题可以设计更多的变种。然而一些调查发现：这些 `LSTM`和`GRU`架构的变种，在广泛的任务中难以明显的同时击败这两个原始架构

## 11. 优化长期依赖

### 11.1 截断梯度

1. 强非线性函数往往倾向于非常大或者非常小幅度的梯度。因此深度神经网络的目标函数经常会存在一个悬崖的地形：宽而且平坦的区域会被目标函数变化快的小区域（悬崖）隔开

   - 这样的问题在于：当参数梯度非常大时，基于梯度下降的参数更新可以将参数投射很远，使得参数进入了一个目标函数较大的区域。这样之前的迭代结果就成了无用功（不管你之前获得了多么小的目标函数，现在统统都被投射到了目标函数较大的区域）
   - 解决的思路是：当梯度较大时，必须使用足够慢的学习率，从而使得更新足够的小
   - 解决方案：梯度截断

2. 梯度截断有两种方案：

   - 在更新参数之前，逐元素的截断参数梯度

   - 在更新参数之前，截断梯度的范数：（其中 $v$ 是梯度范数的上界）

     $$
     \mathbf{\vec g}=\begin{cases}
     \mathbf{\vec g}&, if\; ||\mathbf{\vec g}||<=v\\
     \frac{\mathbf{\vec g}v}{||\mathbf{\vec g}||}&,else
     \end{cases}
     $$

     第二种方案可以确保截断后的梯度仍然是在梯度方向上。但是实现表明：两种方式类似。

3. 当梯度大小超过了阈值时，即使采用简单的随机步骤，效果往往也非常好：

   - 即：随机采用大小为 $v$ 的向量来作为梯度。这样通常会离开数值不稳定的状态

4. 截断一个小批量梯度范数不会改变它的梯度方向。但是许多个小批量使用范数截断梯度之后，它们的平均值并不等同于截断真实梯度

   - 梯度截断使得：小批量梯度下降中，其中真实梯度的方向不再等于所有小批量梯度的平均
   - 这就是说：传统的随机梯度下降使用梯度的无偏估计。而使用范数截断的梯度下降，引入了经验上是有用的启发式偏置

5. 逐元素的梯度截断，梯度更新的方向不仅不再是真实梯度方向，甚至也不是小批量的梯度方向。但是它仍然是一个目标值下降的方向。

### 11.2 引导信息流的正则化

1. 梯度截断有助于处理爆炸的梯度，但是无助于解决梯度消失。为解决梯度消失，有两种思路：

   - 一种思路是：如 LSTM 及其他门控机制，让路径的梯度乘积接近1
   - 另一种思路是：正则化或者约束参数，从而引导信息流

2. 正则化引导信息流：目标是希望梯度向量 $\nabla _{\mathbf{\vec h}^{(t)}}L$ 在反向传播时能维持其幅度。即：希望
   $$
   (\nabla _{\mathbf{\vec h}^{(t)}}L)\frac{\partial \mathbf{\vec h}^{(t)}}{\partial \mathbf{\vec h}^{(t-1)}}
   $$
   与 $\nabla _{\mathbf{\vec h}^{(t)}}L$ 一样大。`Pascanu et al.` 给出了以下正则项：
   $$
   \Omega=\sum_t\left(\frac{||(\nabla _{\mathbf{\vec h}^{(t)}}L)\frac{\partial \mathbf{\vec h}^{(t)}}{\partial \mathbf{\vec h}^{(t-1)}}||}{||\nabla _{\mathbf{\vec h}^{(t)}}L||}-1\right)^{2}
   $$
   - 计算这个正则项可能比较困难。 `Pascanu et al.`提出：可以将反向传播向量 $\nabla _{\mathbf{\vec h}^{(t)}}L$ 考虑作为恒值来近似
   - 实验表明，如果与梯度截断相结合，该正则项可以显著增加 RNN 可以学习的依赖跨度。如果没有梯度截断，则梯度爆炸会阻碍学习的成功

3. 这种方法的一个主要弱点是：在处理数据冗余的任务时，如语言模型，它并不像 LSTM 一样有效


## 12. 外显记忆

1. 神经网络擅长存储隐性知识，但是他们很难记住事实
   - 隐性知识：难以用语言表达的知识。如：狗和猫有什么不同
   - 事实：可以用知识明确表达的。如：猫是一种动物，十月一日是国庆节

2. 虽然随机梯度下降需要都次提供相同的输入，但是该输入也不会被特别精确的存储

3. 神经网络缺乏工作记忆系统，即类似人类为了实现一些目标而明确保存和操作相关信息片段的系统
   - 这种外显记忆组件不仅能让系统快速的存储和检索具体的事实，还能用它们进行循序推论

4. `Weston et al.`引入了记忆网络，其种包括一组可以通过寻址机制来访问的记忆单元

   ![memory_cell](../imgs/10/memory_cell.png)

   - 记忆网络原本需要监督信号来指示它们：如何使用记忆单元
   - `Graves et al.`引入的神经网络图灵机(`Neural Turing machine`:`NTM`)则不需要明确的监督指示采取哪些行动而能够学习从记忆单元读写任意内容，并通过使用基于内容的软注意机制来允许端到端的训练
   - 每个记忆单元都可以被认为是 LSTM 和 GRU 中记忆单元的扩展。不同的是：网络输出一个内部状态来选择从哪个单元读取或者输入
   - 由于产生整数地址的函数非常演绎优化，因此`NTM` 实际上同时从多个记忆单元写入或者读取
     - 在读取时，`NTM`采取多个单元的加权平均值
     - 在写入时，`NTM`同时修改多个单元的值

     用于这些操作的系数被选择为集中于一个小数量的单元（通常采用`softmax`函数产生）。使用这些具有非零导数的权重允许函数控制访问存储器，从而能够使用梯度下降法优化

   - 这些记忆单元通常扩充为包含向量，而不是由 LSTM 或者 GRU 记忆单元所存储的单个标量。原因有两个：
     - 一是：降低读取单个记忆数值的成本。
     - 二是：允许基于内容的寻址

5. 如果一个存储单元的内容在大多数时间步上会被复制（不被忘记），那么它所包含的的信息可以在时间上前向传播，随时间反向传播的梯度也不会消失或者爆炸

6. 外显记忆似乎允许模型学习普通 RNN 或者 LSTM RNN 不能学习的任务

   - 这种优势的一个原因：可能是因为信息和梯度能够在非常长的持续时间内传播

7. 作为存储器单元的加权平均值反向传播的代替，我们可以将存储器寻址稀疏解释为概率，并随机从一个单元读取

   - 但是问题时：优化离散决策的模型需要专门的优化算法
   - 目前为止，训练这些做离散决策的随机架构，仍然要比训练进行软判决的确定性算法更难



