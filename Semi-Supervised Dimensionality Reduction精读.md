# Semi-Supervised Dimensionality Reduction精读

## 摘要

降维在数据挖掘中是一个非常高效的方法。这篇文章研究了一个半监督降维的方法。在这种降维方法中，我们不仅可以利用大量未标记的样本，还可以利用成对约束形式的领域知识，这些约束指定实例对是否属于同一类别（必连约束）或不同类别（勿连约束）。我们提出了SSDR算法，这种算法能够在数据投影到低维空间后仍保持数据的内在结构，同时保持在标记示例上定义的必要链接和非必要链接约束。SSDR算法非常高效并且有闭式解。SSDR算法在大量数据集上都要优于许多已有的降维方法。

## 介绍

随着图像数据、金融时间序列数据和基因数据等高维数据的迅速累积，降维成为了许多数据挖掘任务的基础工作。根据数据是否有标记，现有的降维算法可以粗略的分为有监督的和无监督的。fisher线性判别法是有监督降维算法的一个例子，它可以在有标签的数据中提取最佳判别向量。而主成分分析法是无监督降维方法的一个例子，它的工作原理是在无法获得类标签的情况下，尝试保留数据的全局协方差结构。

半监督降维方法可以被视作是半监督学习的一个新的问题，半监督学习是从带标签的数据和不带标签的数据中组合学习的。在许多实际的数据挖掘任务中，不带标签的数据比带标签的数据更容易获得，因此半监督学习现在是一个备受关注的算法。现今的半监督学习可以粗略的分为三类，分别是半监督分类，半监督回归，半监督聚类。

在许多数据挖掘任务中，利用领域知识一直是一个重要问题。一般来说，领域知识可以以多种形式表达，如类标签、成对约束或其他先验信息。我们重点关注那些成对约束出现的领域知识，即已知属于同一类别（必连约束）或不同类别（勿连约束）的成对实例。成对约束在许多任务中都有使用，例如图像检索任务中。在这些场景的应用中，成对约束比获取类别标签更实用，因为真实标签不一定是已知的，而让一些用户辨别一些实例是否属于同一类别可能更为容易。此外成对约束可以从有标签的数据中推测出来，也可以从有标签的数据中推断出成对约束。此外，不像数据的分类标签，成对约束可以不用人工标记，可以自动获得。

“常见的先验知识主要有两种。第一种是标号信息，这类先验信息用在分类方面比较多，如Cai deng的SDA，Zhou deng yong的Label Propagation等； 
第二种是成对约束（pairwise constraint），这类信息主要用于聚类，约束有两种，一个是正约束(must-link)，正约束指定两个sample必须属于同一类；另一个是 负约束(cannot-link)，与正约束相反。这两种信息都可以结合进半监督降维中。不过约束形式的先验知识更普遍，应用更广。主要原因有2个：1是 成对约束比标号信息更容易获得，人工标注需要相关领域的专家知识，而分辨两个样本是否属于同一类则轻松的多，甚至普通人就能做到；2是从标号信息可以推出 约束信息，反之则不然，因此可以说约束比标号更普遍、通用” ([“Semi-Supervised Dimensionality Reduction - Hiroki - 博客园”](zotero://select/library/items/5V35C723)) ([snapshot](zotero://open-pdf/library/items/39PAWNBR?sel=%23cnblogs_post_body))

## 相关研究

1. “Bar-Hillel et al.” ([Zhang 等, 2007, pp. -](zotero://select/library/items/H3LCMTLM)) ([pdf](zotero://open-pdf/library/items/WTDUREQR?page=1)) 提出了cFLD，作为RCA的中间步骤，用于从等价约束中降维。但是cFLD只能解决一些必连约束问题。在约束有限时存在奇异问题。
2. “Tang and Zhong [15]” ([Zhang 等, 2007, pp. -](zotero://select/library/items/H3LCMTLM)) ([pdf](zotero://open-pdf/library/items/WTDUREQR?page=1)) 在降维中使用了成对约束，同时利用了必连约束和勿连约束，但是他们并没有考虑到那些没有标记的数据。
3. “Yang et al.” ([Zhang 等, 2007, pp. -](zotero://select/library/items/H3LCMTLM)) ([pdf](zotero://open-pdf/library/items/WTDUREQR?page=1)) 利用数据样本的对曲面坐标这些先验信息进行降维。显然坐标信息并没有成对约束这么好获取。

在本文中，我们研究了同时存在未标记数据和成对约束条件的降维问题。我们提出了一个简单并且高效的算法，SSDR。这个算法可以同时保留原始高维数据的结构信息和用户指定的配对约束信息。此外，SSDR 对某些特定拉普拉卡矩阵的特征问题有一个闭式解，因此它非常高效。

## SSDR算法介绍

目标函数

$$
\begin{aligned}J(\boldsymbol{w})&=\quad\frac1{2n_C}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in C}(y_i-y_j)^2\\&-\frac\beta{2n_M}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in M}(y_i-y_j)^2\\&=\quad\frac1{2n_C}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in C}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\\(2.1)&-\frac\beta{2n_M}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in M}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\end{aligned}
$$
我们需要最大化$J(w)$使得勿连约束之间的距离越来越大，必连约束之间的距离越来越小。但是这个目标函数只考虑了数据之间的成对约束，并没有考虑那些未标记的数据（数据即无成对约束也没有类别标签）。

优化后的目标函数为：

$$
\begin{aligned}J(\boldsymbol{w})&=\quad\frac1{2n^2}\sum_{i,j}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\\&+\frac\alpha{2n_C}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in C}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\\(2.2)&-\frac\beta{2n_M}\sum_{(\boldsymbol{x}_i,\boldsymbol{x}_j)\in M}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\end{aligned}
$$
加入未标记数据的原因是提高在没有成对约束时模型的表征能力。式$(2.2)$的首相表示在降维后的空间中所有样本之间的距离平方（该项能让降维后的数据尽可能分散），并且该项只有在有标记数据数量很少的时候起决定性作用，其中$\alpha,\beta$ 为缩放系数。通常勿连约束点对之间的距离要比其它两项大的多，因此我们要调节这三项之间的比例，使得目标函数不会失衡，一般$\alpha$要小一些$\beta$要大一些。

当$\alpha,\beta$ 取值很大时式$(2.2)$ 将退化为式$(2.1)$ 该式子的简要表达形式为：

$$
\begin{aligned}&(2.3)&&J(\boldsymbol{w})=\frac{1}{2}\sum_{i,j}(\boldsymbol{w}^{T}\boldsymbol{x}_{\boldsymbol{i}}-\boldsymbol{w}^{T}\boldsymbol{x}_{\boldsymbol{j}})^{2}\boldsymbol{S}_{\boldsymbol{i}\boldsymbol{j}}\\&\text{where}\\&(2.4)&&S_{ij}=\left\{\begin{array}{ll}\frac{1}{n^2}+\frac{\alpha}{n_G}&\mathrm{if}\left(x_i,x_j\right)\in C\\\frac{1}{n^2}-\frac{\beta}{n_M}&\mathrm{if}\left(x_i,x_j\right)\in M\\\frac{1}{n^2}&\mathrm{otherwise}\end{array}\right.\end{aligned}
$$
 上述公式经过推导有：

$$
\begin{aligned}
&&&\frac12\sum_{i,j}(\boldsymbol{w}^T\boldsymbol{x}_i-\boldsymbol{w}^T\boldsymbol{x}_j)^2\boldsymbol{S}_{ij} \\
&\text{=}&& \frac12\sum_{i,j}(\boldsymbol{w}^T\boldsymbol{x}_i\boldsymbol{x}_i^T\boldsymbol{w}+\boldsymbol{w}^T\boldsymbol{x}_j\boldsymbol{x}_j^T\boldsymbol{w}-2\boldsymbol{w}^T\boldsymbol{x}_i\boldsymbol{x}_j^T\boldsymbol{w})\boldsymbol{S}_{ij} \\
&\text{=}&& \sum_{i,j}\boldsymbol{w}^T\boldsymbol{x}_i\boldsymbol{S}_{ij}\boldsymbol{x}_i^T\boldsymbol{w}-\sum_{i,j}\boldsymbol{w}^T\boldsymbol{x}_i\boldsymbol{S}_{ij}\boldsymbol{x}_j^T\boldsymbol{w} \\
&\text{=}&& \sum_i\boldsymbol{w}^T\boldsymbol{x}_iD_{i\boldsymbol{i}}\boldsymbol{x}_i^T\boldsymbol{w}-\boldsymbol{w}^T\boldsymbol{X}\boldsymbol{S}\boldsymbol{X}^T\boldsymbol{w} \\
&\text{=}&& w^TX(D-S)\boldsymbol{X}^T\boldsymbol{w} \\
&=\quad\boldsymbol{w}^T\boldsymbol{X}\boldsymbol{L}\boldsymbol{X}^T\boldsymbol{w}
\end{aligned}
$$
其中$D_{ii}$ 是一个对角矩阵，因为$\sum_{i,j}w^Tx_iS_{ij}x_i^Tw$  是一个只含平方项的二次型，$L$ 为拉普拉斯矩阵。

显然我们可以通过计算$XLX^T$ 的前n个最大特征值求解式的最大值。

## SSDR于拉普拉斯方法的区别

1. 拉普拉斯方法源于无监督降维，而SSDR用于半监督降维。
2. 拉普拉斯方法构建邻接矩阵采用的是k邻接方法，而SSDR采用的是成对约束，直接来源于式（2.4）。

## 实验