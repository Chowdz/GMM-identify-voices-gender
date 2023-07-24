# GMM-identify-voices-gender
EM algorithm based on GMM to extract voice features and identify male and female voices

对于样本 $x\sim F(x,\theta)$，有$x=(x^{(1)},x^{(2)},...,x^{(n)})$，若用$MLE$方法求未知参数$\theta$，则

$\hat{\theta}=\mathop{argmax}\limits_{\theta}L(\theta)=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n lnp(x^{(i)},\theta)$，但如果样本来自若干个仅参数不同的相同分布，就无法用$MLE$方法求出，可以设每

个样本$x^{(i)}$有$z^{(k)}$的概率服从于某个分布，$z^{(k)}$也是个随机变量，服从于多项分布，

有$z=(z^{(1)},z^{(2)},...,z^{(k)})$，设$P(z^{(i)})=Q_i(z^{(i)})$，

那么有$\sum_{z^{(i)}}Q_i(z^{(i)})=1$，且对于求解$\theta$来说，$z$是隐变量，那么如何求解未知参数$\theta$呢？有：

$\hat{\theta}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n lnp(x^{(i)},\theta)=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}P(z^{(i)})p(x^{(i)},\theta|z^{(i)})] \\ =\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)]$ 

那么将$\hat{\theta}$作变换，有$\hat{\theta}=\mathop{argmax}\limits_{\theta}\sum_{i=1}^n ln[\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]$ ，由$Jensen$不等式可知，若$f^{''}(x)\geq0$，有$Ef(x)\geq f(Ex),\because(lnx)^{''}=-\frac{1}{x^2}<0,\therefore Elnx \leq ln(Ex)$，可以看出$\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$的关于$z^{(i)}$的期望。



$\therefore\sum_{i=1}^n ln[\sum_{z^{(i)}}Q_i(z^{(i)})\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]=\sum_{i=1}^n ln[E\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]\geq\sum_{i=1}^n Eln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})} \\ =\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})ln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})} \\ =\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)-\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnQ_i(z^{(i)})$



$\therefore \hat{\theta}=\mathop{argmax}\limits_{\theta} \Big[\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)-\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnQ_i(z^{(i)})\Big]$，可以看出上式后半部分与$\theta$无关

$\therefore \hat{\theta}=\mathop{argmax}\limits_{\theta} \sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})lnp(x^{(i)},z^{(i)},\theta)$

但注意，我们这里是当且仅当$Jensen$不等式成立时，也就是$P(X=EX)=1$，即认为$X$为常数时，放到这里也就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$（$c$为常数），$\because L(\theta)=\sum_{i=1}^n ln[E\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}]\geq\sum_{i=1}^n Eln\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}$，若我们最大化$L(\theta)$，通过最大化该式右半部分也就是最大化$L(\theta)$的下界，未必可以找到$L(\theta)$的最大值，除非等号成立，也就是$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，这其实就是$EM$算法中的$E$步。

若定义$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，$\therefore p(x^{(i)},z^{(i)},\theta)=c\ast Q_i(z^{(i)})$，两边对$z^{(i)}$求和，$\therefore \sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)=c\ast \sum_{z^{(i)}}Q_i(z^{(i)})=c$

$\therefore \frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta) ,\therefore Q_i(z^{(i)})=\frac{p(x^{(i)},z^{(i)},\theta)}{\sum_{z^{(i)}}p(x^{(i)},z^{(i)},\theta)}=\frac{p(x^{(i)},z^{(i)},\theta)}{\sum_{z^{(i)}}p(z^{(i)})p(x^{(i)},\theta|z^{(i)})}=p(z^{(i)}|x^{(i)},\theta)$

$\therefore$当且仅当$\frac{p(x^{(i)},z^{(i)},\theta)}{Q_i(z^{(i)})}=c$，即为定值时，有：

<font color="#dd000"> $\hat{\theta}=\mathop{argmax}\limits_{\theta} \sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta)*lnp(x^{(i)},z^{(i)},\theta)$</font>

> 可以看出$\sum_{i=1}^n \sum_{z^{(i)}}p(z^{(i)}|x^{(i)},\theta)*lnp(x^{(i)},z^{(i)},\theta)$就是$lnp(x^{(i)},z^{(i)},\theta)$关于$z|x,\theta$的条件期望，也就是当样本和未知参数$\theta$给定时，我们既然不知道隐变量$z$的取值，就用期望去替代它，当然时在样本$x$和参数$\theta$给定的条件下，求完这个条件期望，再去关于$\theta$去极大化$L(\theta)$，这就是$EM$算法的基本思想。
>
> 所以，我们的$E$步就是，先随机定出$\theta^{j}$作为初始值，当然这个初始值也包含隐变量$z$，然后利用样本$x$去求条件概率$Q_i(z^{(i)})=p(z^{(i)}|x^{(i)},\theta^j),j=1,2,...N$为迭代次数，$\theta^j$和$\theta$不一样，$\theta$是上帝知道的最优值，$\theta^j$是不断迭代去靠近$\theta$的值，求出$Q_i(z^{(i)})$后代入到$L(\theta)=\sum_{i=1}^n \sum_{z^{(i)}}Q_i(z^{(i)})*lnp(x^{(i)},z^{(i)},\theta)$当中，然后对$\theta$求偏导，极大化$L(\theta)$，得到$\theta^{j+1}$，这是$M$步。再把$\theta^{j+1}$替换原来的$\theta^j$，再重新执行$E$步，不断迭代，直到$L(\theta)$收敛。
