### NSLS: Nondominated sorting and local search based algorithm

##### Reference: Chen B, Zeng W, Lin Y, et al. A new local search-based multiobjective optimization algorithm[J]. IEEE Transactions on Evolutionary Computation, 2014, 19(1): 50-73.

##### NSLS is a multi-objective evolution algorithm (MOEA) based on local search, nondominated sorting, and the farthest candidate method.

| Variables | Meaning                                                      |
| --------- | ------------------------------------------------------------ |
| npop      | Population size                                              |
| iter      | Iteration number                                             |
| lb        | Lower bound                                                  |
| ub        | Upper bound                                                  |
| mu        | The mean value of the Gaussian distribution (default = 0.5)  |
| delta     | The standard deviation of the Gaussian distribution (default = 0.1) |
| nvar      | The dimension of decision space                              |
| pop       | Population                                                   |
| objs      | Objectives                                                   |
| new_pop   | The newly generated population                               |
| new_objs  | The objectives of the newly generated population             |
| dom       | Domination matrix                                            |
| pf        | Pareto front                                                 |

#### Test problem: ZDT3



$$
\left\{
\begin{aligned}
&f_1(x)=x_1\\
&f_2(x)=g(x)\left[1-\sqrt{x_1/g(x)}-\frac{x_1}{g(x)}\sin(10\pi x_1)\right]\\
&f_3(x)=1+9\left(\sum_{i=2}^nx_i\right)/(n-1)\\
&x_i\in[0, 1], \qquad i=1,\cdots,n
\end{aligned}
\right.
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 250, np.array([0] * 30), np.array([1] * 30))
```

##### Output:

![](https://github.com/Xavier-MaYiMing/NSLS/blob/main/Pareto%20front.png)

```python
Iteration 20 completed.
Iteration 40 completed.
Iteration 60 completed.
Iteration 80 completed.
Iteration 100 completed.
Iteration 120 completed.
Iteration 140 completed.
Iteration 160 completed.
Iteration 180 completed.
Iteration 200 completed.
Iteration 220 completed.
Iteration 240 completed.
```

