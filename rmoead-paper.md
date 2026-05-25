A reinforcement learning based RMOEA/D for bi-objective fuzzy flexible job
shop scheduling

∗ ∗
Rui Li, Wenyin Gong, Chao Lu

School of Computer Science, China University of Geosciences, Wuhan 430074, China

A R T I C L E I N F O

| Dataset link: [https://cuglirui.github.io/downloads.htm](https://cuglirui.github.io/downloads.htm) |
| --- |
| Keywords: |
| Fuzzy flexible job shop scheduling |
| Multi-objective optimization |
| Parameter adaption |
| Reinforcement learning |
| MOEA/D |

A B S T R A C T

The flexible job shop scheduling problem (FJSP) is significant for realistic manufacturing. However, the
job processing time usually is uncertain and changeable during manufacturing. This paper presents a multiobjective FJSP with fuzzy processing time (MOFFJSP) for optimizing the makespan and total machine workload
as objectives. To solve the MOFFJSP, a MOEA/D based on reinforcement learning named RMOEA/D is
proposed. RMOEA/D can be featured as: (i) an initial strategy with three rules is used to get a high-quality
initial population; (ii) a parameter adaption strategy based on Q-learning is proposed to guide the population
choose the best parameter to increase diversity; (iii) a variable neighborhood search based on reinforcement
learning is designed to lead the solution to choose the right local search method; and (iv) an elite archive
is used to improve the usage rate of the abandoned historical solution. RMOEA/D is compared with five
well-known realted methods, i.e., MOEA/D, NSGA-II, MOEA/D-M2M, NSGA-III and IAIS on three benchmark
suites. The results show that RMOEA/D outperforms these five state-of-art algorithms.

to efficiently solve FFJSP.
Furthermore, with the development of intelligent manufacturing
and industrial 4.0, many industrial start to consider the low energy consumption manufacturing (Lu et al.,2021). Many energy-aware scheduling models have been proposed including distributed hybrid flow

1. Introduction

1.1. Background

With the development of economic globalization, traditional flexible manufacturing met a quite huge challenge. And it is difficult
to satisfy the requirement of the market (Lang et al.,2021;Rifai
et al.,2021);. Flexible job shop scheduling problem (FJSP) is a classical scheduling problem has been intensively studied over the past
decades. Many heuristic algorithms have been proposed for FJSP such
as genetic algorithm (GA) (Yuan et al.,2020), artificial bee colony
algorithm (ABC) (Li, Huang, et al.,2020), two-phase meta-heuristic (Lei
et al.,2019), Jaya (Caldeira & Gnanavelbabu,2021), and teaching–
learning-based optimization (TLBO) (Lei et al.,2018). However, fixed
processing time is too idealized to simulate the practical manufacturing.
In practical flexible manufacturing, the processing time is uncontrollable and it floats between an interval (Pan et al.,2021;Zhu & Zhou,
2021). So it is necessary to fuzzy the processing time for FJSP. FJSP
with fuzzy processing time (FFJSP) is an extension of FJSP. FJSP has
been proved as an NP-hard problem and FFJSP is also an NP-hard
problem (Pavlov et al.,2019). Moreover, it is worth for studying how
to efficiently solve FFJSP.
Furthermore, with the development of intelligent manufacturing

Inspired by scalar objective optimization problem,Zhang and Li
(2007) proposed multi-objective evolutionary algorithm based on decomposition (MOEA/D) for multi-objective optimization problem
(MoP). Based on reference vectors and the Tchebicheff function,
MOEA/D can get good convergence and perform well diversity simultaneously. Recently, reinforcement learning (RL) is truing into a hot
topic due to its strong ability of decision and optimization. Cooperation
between the RL and evolutionary algorithms is a promising direction to
solve complex optimization problems (Gong et al.,2021;Shiue et al.,
2018;Zhao et al.,2019). Facing the complex optimization problems, it
is expected to achieve superior performance via the synergy between RL
and evolutionary computation as well as the problem-specific operators
within the MOEA/D framework.

∗ Corresponding authors.
E-mail addresses: [liruicug@163.com](mailto:liruicug@163.com)(R. Li), [wygong@cug.edu.cn](mailto:wygong@cug.edu.cn)(W. Gong), [luchao@cug.edu.cn](mailto:luchao@cug.edu.cn)(C. Lu).

E-mail addresses: [liruicug@163.com](mailto:liruicug@163.com)(R. Li), [wygong@cug.edu.cn](mailto:wygong@cug.edu.cn)(W. Gong), [luchao@cug.edu.cn](mailto:luchao@cug.edu.cn)(C. Lu).

[https://doi.org/10.1016/j.eswa.2022.117380](https://doi.org/10.1016/j.eswa.2022.117380)
Received 15 September 2021; Received in revised form 13 March 2022; Accepted 25 April 2022

Received 15 September 2021; Received in revised form 13 March 2022; Accepted 25 April 2022
$YDLODEOH RQOLQH 7 May 2022
0957-4174/© 2022 (OVHYLHU Ltd. All ULJKWV UHVHUYHG

0957-4174/© 2022 (OVHYLHU Ltd. All ULJKWV UHVHUYHG

* * *

1.2. Motivations

The MOFFJSP has been extensively studied for several years. However, in the existing literature, most methods execute local search
strategies by poll pattern, which is inefficient and blind. Moreover, the
performance of the algorithm is limited to parameter selection. A lot of
black-box testing is performed to find the best parameter. Parameter
selection problem lacks prior knowledge and it is time-consuming.
To optimize makespan and total machine workload simultaneously, it
is challenging and significant to develop effective algorithms for the
MOFFJSP. Motivated by the above problems, this paper proposed an
RL-based MOEA/D (RMOEA/D), which contains the following innovations: First in order to guide each solution adaptively to select the
best local search strategy, a variable neighborhood search based on RL
(RVNS) is proposed. Second, to make MOEA/D automatically adjust
the parameter T, a parameter selection strategy based on Q-learning
(Q-PAS) is designed. Next, an initial method integrating a variety of
initial strategies is designed to get a high convergence and diverse
population. Then, a discrete crossover and mutation method is used to
obtain a large searching step. Moreover, an elite archive is applied to
improve the utilization rate of abandoned solutions. Finally, to verify
the performance of RMOEA/D, extensive numerical tests are carried out
and the comparative results demonstrate the effectiveness of the above
designs and the superiority of the proposed algorithm in solving the
MOFFJSP.

1.3. Contributions

The main contributions of this paper go in five directions.

(1)An initial strategy which combines the advantage of the three
strategies is designed to provide an initial population with high
convergence and diversity.
(2)A parameter selection strategy based on Q-learning is proposed

(2)A parameter selection strategy based on Q-learning is proposed
to let MOEA/D automatically select the best T to improve the
diversity of PF.
(3)A variable neighborhood search method based on RL is proposed

diversity of PF.
(3)A variable neighborhood search method based on RL is proposed
to efficiently execute several local search strategies.
(4)An elite archive is designed to collect the historical elite solution

(4)An elite archive is designed to collect the historical elite solution
to increase the utilization rate of abandoned solutions.
(5)The performance of RMOEA/D is executed on 23 FFJSP in-

(5)The performance of RMOEA/D is executed on 23 FFJSP instances with different features. Experimental results show that
RMOEA/D is superior to state-of-art algorithms under the condition of faster convergence.

The rest of this paper is organized as follows. Section2illustrates
the literature review and some basic concept of triangular fuzzy processing time. In Section3, the problem statement and modeling are
introduced. In Section4, the details of our approach RMOEA/D are
reported. Numerical test experiments on RMOEA/D and discussion are
shown in Section5and conclusions are summarized in Section6.

2. Related work and background knowledge

search strategy and utilize the hyper-level strategy to manipulate the
low-level heuristics to obtain a good performance than other existing
algorithms.Dorfeshan et al.(2020) abstract fuzzy processing time as
a decision-maker and propose a weight distance approximate method,
which can determine operation sequence better.Li, Liu, et al.(2020)
propose a Type-2 fuzzy processing time to handle high uncertainty
of complexity system, which supplements the disadvantage of the
traditional triangular fuzzy number. Moreover, automatically adjusting
population size is an efficient technique, but has not been applied
to MOFFJSP.Pan et al.(2021) propose a bi-population evolutionary
algorithm with feedback schema, which can make up for that gap.
However, to the best of our knowledge, MOFFJSP has seldom been
studied. So it is significant to design an algorithm for MOFFJSP.

The MOEA/D is a classical multi-objective evolutionary algorithm
(MOEA) that has been successfully applied for a variety of scheduling
problems such as permutation flow shop scheduling problem (Pericleous et al.,2017), satellite range scheduling problem (Du et al.,
2019), optimal power flow problem (Zhang et al.,2020), distributed
heterogeneous hybrid flow shop scheduling problem (Shao et al.,
2021a), and distributed heterogeneous welding flow shop scheduling
problem (Wang et al.,2021). Referring to the superior performance
of MOEA/D, the main motivations of using MOEA/D are following:
(1) MOEA/D (Zhang & Li,2007) has a great diversity to solve MoP.
(2) RL can make MOEA/D select the best parameter and local search
strategy which is more efficient than randomly selection. Thus, this
paper applied an RL-based MOEA/D for MOFFJSP.

2.2. MOEA/D

2.3. Reinforcement learning applied to scheduling

Recently, as a very popular artificial intelligence approach, RL has
been widely applied in shop scheduling problems.Qu et al.(2015) developed a centralized RL approach for scheduling multi-state processes
and multiple machines manufacturing systems.Qu et al.(2016) also designed a RL-based sched-uling approach. It can adaptively update production plans by adopting real-time product and processing events information during executions.Wang and Yan(2016) proposed a knowledgeable multi-agents-based self-adaptive scheduling and adopted a
dynamic scheduling strategy based on weighted Q-learning. To improve the performance,Shahrabi et al.(2017) put forward a Q-factor
algorithm for parameter estimation to solve dynamic JSP with random
job arrivals and machine breakdowns.Zhang et al.(2017) proposed
simulation-based Q-learning for the scheduling problem from a Markov
decision process perspective.Palombarini and Martínez(2018) adopted
a deep RL approach that preserved the rescheduling knowledge in the
deep Q-network in order to learn schedule repairing policies straightly
from high-dimensional sensory inputs.Waschneck et al.(2018) applied
a deep Q-network to semiconductor manufacturing scheduling and
trained a deep neural network with flexible objectives.Shiue et al.
(2018) designed a RL based approach including an off-line learning module and Q-learning. This method performs better than the
previously dispatching rules.Ahmadi et al.(2018) modeled the job
sequencing problem as a traveling salesman problem of second-order
and solved it with a dynamic Q-learning-based genetic algorithm.Zhao
et al.(2019) adopted a double-layer action Q-learning algorithm to
solve dynamic FJSP and showed that this approach was suitable for
dynamic FJSP.Palombarini and Martínez(2019) also applied the realtime rescheduling task as a closed-loop control problem and trained
a deep Q-network to select repair actions in response to unexpected
events and disturbances.Lin et al.(2019) proposed an edge computingbased smart manufacturing factory framework and used an improved
deep Q-network to solve the JSP.Han and Yang(2020) designed
a dueling double deep Q-network which combined the advantages
of real-time response and flexibility of a deep convolutional neural

* * *

Table 1
Literature review of RL applied to scheduling.
References Type of problem

| References | Type of problem | Objective number | Machine number | Method |
| --- | --- | --- | --- | --- |
| (Qu et al.,2015) | Dynamic job shop scheduling | S | M | Centralized RL |
| (Qu et al.,2016) | Dynamic job shop scheduling | S | M | RL |
| (Wang & Yan,2016) | Knowledgeable manufacturing system | S | M | Weighted Q-learning |
| (Shahrabi et al.,2017) | Dynamic job shop scheduling | S | M | Q-factor algorithm |
| (Zhang et al.,2017) | Real-time job shop scheduling | S | M | Markov decision processes |
| (Palombarini&Martínez,2018) | Socio-technical manufacturing system | S | M | Deep Q-learning |
| (Waschneck et al.,2018) | Job shop scheduling | S | M | Deep Q-network |
| (Shiue et al.,2018) | Real-time job shop scheduling | S | M | RL |
| (Ahmadi et al.,2018) | Job sequencing and tool switching problem | S | S | Q-learning |
| (Zhao et al.,2019) | Dynamic job shop scheduling | S | M | Q-learning |
| (Palombarini&Martínez,2019) | Semicondutor production scheduling | S | M | Deep Q-network |
| (Lin et al.,2019) | Socio-technical manufacturing system | S | M | Deep Q-network |
| (Han&Yang,2020) | Job shop scheduling | S | M | Dueling Double Deep Q-network |
| (Luo,2020) | Dynamic job shop scheduling | S | M | Deep Q-learning |
| (Wang,2020) | Dynamic job shop scheduling | M | M | Weighted Q-learning |
| (Liu et al.,2020) | Job shop scheduling | S | M | Actor-Critic DRL |

A fuzzy set F̃ contains two elements including x and membership
function 𝜇F̃(x). 𝜇F̃(x) is the possibility of x belonging to F̃. All of x
belong to a definite set X. The definition of fuzzy set is given as follows:

network and RL and learned behavior strategies directly according to
the input manufacturing states.Luo(2020) adopted deep Q-learning
with a continuous state for dynamic FJSP, and it is superior to other
dispatching rules.Wang(2020) proposed an adaptive strategy for FJSP
with a weighted Q-learning algorithm by clustering to dynamically find
the most suitable operation.Liu et al.(2020) viewed the JSP as a
sequential decision-making problem and proposed deep RL to deal with
it. To speed up model training, they also proposed a parallel training
method that combined asynchronous updates with deep deterministic
policy gradient.Table1showed literature summary about JSP with RL.
Although RL techniques are widely applied in manufacturing, it has

\\mu\_{\\tilde{F}}(x).\ \\mu\_{\\tilde{F}}(x)

2.4. Fuzzy Set

\\tilde{F}

\\widetilde{v}=\\left{x,\\mu\_{\\bar{F}}(x)\|\\forall x\\in X\\right},0\\leqslant\\mu\_{\\bar{F}}(x)\\leqslant

(1)

The classical set is definite. The fuzzy set transfer to classical set,
when 𝜇F̃(x) = 1,
{}

2.5. Fuzzy operators

1 2 3 1 2 3
(1) Addition operator: ̃s + t̃ = (s1+ t1,s2+ t2,s3+ t3).

The most classical membership function in scheduling is triangle
fuzzy number (TFN). As shown inFig.1, the membership function is
similar to the triangle. t1is the earliest processing time, t2is the most
possible processing time and t3is the latest processing time. A triple
(t1,t2,t3) is a TFN to represent the processing time. The formulation of
the membership function is given as below:
⎧0, x ⩽ t,

\\tilde{t}=(t\_{1},t\_{2},t\_{3})

t\_{1}

\\mu\_{\\tilde{F}}(x)=1

\\mu\_{\\tilde{F}}(x)=\\left{\\begin{array}{l l}{0displaystyle,qquad\ x\\leqslant t\_{1},}\ {\\displaystyle\\frac{x-t\_{1}}{t\_{2}-t\_{1}},,t\_{1}<x\\leqslant t\_{2},}\ {\\displaystyle\\frac{t\_{3}-x}{t\_{3}-t\_{2}},,t\_{2}<x<t\_{3},}\ {0\\\ \ \ \ \ x\\geqslant t\_{3}.}\\end{array}\\right.

(t\_{1},t\_{2},t\_{3})

\\tilde{\\mathfrak{s}}+\\tilde{\\mathfrak{t}}=(\\mathfrak{s} _{1}+t_{1},\\mathfrak{s} _{2}+t_{2},\\mathfrak{s} _{3}+t_{3}).

Fig. 1. Triangular membership function.

f\_{2}(\\tilde{s}),>,f\_{2}(\\tilde{t}),

x1 +2x2 +x3
(2) Ranking operator: (a) f ( ̃x) =1. if f ( ̃s) > f (t̃),1 1̃s > t̃,
4
otherwise ̃s < t̃. (b) f ( ̃x) = x2 2, when f1( ̃s) = f (t̃1), if f (2̃s) > f (t̃),2
then ̃s > t̃; otherwise ̃s < t̃. (c) f ( ̃x) = s3 3− s , when1f (2̃s) = f (t̃2), if
f3( ̃s) > f3(t̃), then ̃s > t̃; otherwise ̃s < t̃.
(3) Max operator: if ̃s > t̃, then ̃s ∨ t̃ = ̃s; otherwise ̃s ∨ t̃ = t̃.

f\_{1}(\\tilde{s})>f\_{1}(\\tilde{t}),,,\\tilde\ s>\\tilde{t}

\\begin{array}{r}{f\_{1}(\\tilde{x})=\\frac{x\_{1}+2x\_{2}+x\_{3}}{4}}\\end{array}

f\_{2}(\\tilde{s})=f\_{2}(\\tilde{t})

f\_{2}(\\tilde{x}),=,x\_{2}

\ !f\_{1}({\\tilde{s}}),=,f\_{1}({\\tilde{t}}),

\\tilde{s},<,\\tilde{t}.

\\varrho\_{i}

\\tilde{s}\\vee\\tilde{t}=\\tilde{t}.

f\_{3}({\\tilde{x}})=s\_{3}-s\_{1}

\\tilde{s}>\\tilde{t};

\\tilde{s}<\\tilde{t}.

• No jobs may be processed on more than one machine at a time.
• The operation cannot be interrupted and the machine break down

3
(3) Max operator: if ̃s > t̃, then ̃s ∨ t̃ = ̃s; otherwise ̃s ∨ t̃ = t̃.

\\tilde{\\mathfrak{H}}\\vee\\tilde{t}=\\tilde{\\mathfrak{H}};

3. Problem statement and mathematical modeling

f\_{3}(\\tilde{s})>f\_{3}(\\tilde{t}),

\\tilde{s}>\\tilde{t};

\\mathcal{J}={\\mathcal{J} _{1},\\mathcal{J}_{2},\\ldots,\\mathcal{J} _{i},,\\ldots,\\mathcal{J}_{n}}

A flexible job shop scheduling problem with fuzzy processing time
from a real-world manufacturing process can be described as follow.
 = { ,  , …, 1 2 i, …,  } is the job set and  = { ,  , …,  ,n 1 2 k
…, m} is the machine set. Each job ihas a set of 𝛩ioperations,
Oi= {Oi,1,Oi,2, …,Oi,j, … ,Oi,i}, Oi,j∈ O . Each operation can bei
processed on part of machines or all machines. And the processing
time is a TFN P̃i,j,k= (p1,p2,p3). FFJSP includes two subproblems,
machine assignment and operation sequencing. The former is that each
operation selects a machine from a candidate set. The latter is to
schedule all operations on all machines to get feasible schedules. Some
assumptions of FFJSP are given below:

• No jobs may be processed on more than one machine at a time.
• The operation cannot be interrupted and the machine break down
is not considered.
• Setup times and removal times are included in processing time

O\_{i} ~~=~~{O\_{i,1},O\_{i,2},\\ldots,O\_{i,j},\\ldots ~~,O\_{i,\\varTheta\_{i}}},~~ O\_{i,j} ~~\\in~~ O\_{i}.~\

• Setup times and removal times are included in processing time
• Only if the previous operation of each job has been finished, the

\\ldots,\\mathcal{M}\_{m}}

• Setup times and removal times are included in processing time
• Only if the previous operation of each job has been finished, the
next operation can be handled.
• The processing time of each job is a TFN.

* * *

3.2. Notations

n: The number of jobs.
m: The number of machines.

m: The number of machines.
i: The index of jobs.

i: The index of jobs.
j: The index of operations.

k: The index of machines.
𝛩 : The total operation number of job 

j: The index of operations.
k: The index of machines.

𝛩i: The total operation number of job i.
O : The j operation of job , j = 1, …,𝛩

\\mathcal{J}\_{i}

\\Theta\_{i}.

i i
Oi,j: The jtℎoperation of job i, j = 1, …,𝛩i.
S̃ : The beginning time of operation O, which is a TFN: S̃

O\_{i,j}

i,j tℎ i i
S̃i,j: The beginning time of operation Oi,j, which is a TFN: S̃i,j=
(s1,s2,s3).
C̃ The finish time of operation O, which is a TFN: C̃ =

j\_{t h}

\\mathcal{J} _{i},,j=1,\\ldots,\\theta_{i}

O\_{i,j},

\\tilde{S}\_{i,j}\

\\mathrm{T F N};\\tilde{S}\_{i,j}=

(s\_{1},s\_{2},s\_{3})

\\tilde{C}\_{i,j}

O\_{i,j}

\\tilde{C}\_{i,j}\ =

(c\_{1},c\_{2},c\_{3})

1 2 3
P̃i,j,k: The processing time of jtℎoperation of job ion machine k,
which is a TFN: P̃i,j,k= (p1,p2,p3)
x : If the j operation of job  is processed on machine , the

\\tilde{P}\_{i,j,k},

\\mathcal{I}\_{i}

j\_{t h}

\\tilde{P} _{i,j,k}=(p_{1},p\_{2},p\_{3})

\\mathbf{x}\_{i,j,k},

\\mathcal{I}\_{i}

\\mathcal{M}\_{k}

j\_{t h}

value is set to 1; otherwise is set to 0.
ui1,j1,i2,j2:If the j2tℎoperation of job i2is processed on machine k
and machine kis also the candidate machine of the j1tℎoperation of
job i1, the value is set to 1; otherwise is set to 0.

\\mathbf{u} _{i_{1},j\_{1},i\_{2},j\_{2}}\\mathrm{; ~~f f~~

\\mathcal{I} _{i_{2}}

j\_{2t h}

j\_{1t h}

\\mathcal{M}\_{k}

\\mathcal{I}\_{i,1

The mathematical model of MOFFJSP is given as below:
{}

3.3. The mathematical model of the MOFFJSP

m i n F\_{1}=\\operatorname\*{m a x}\\left{\\tilde{C} _{i,t_{i}}\\right},1\\leq i\\leq n.

m i n F\_{2}=\\sum\_{k=1}^{m}W\_{k}.

(5)

(7)

W\_{k}=\\sum\_{i=1}^{n}\\sum\_{j=1}^{\\theta\_{i}}\\tilde{P} _{i,j,k}\*\\mathbf{x}_{i,j,k}.

\\tilde{C} _{i,j}=\\tilde{S}_{i,j}+\\tilde{P}\_{i,j,k}.

(8)

(10)

(begin{array}{r}{(\\tilde{S} _{i_{1},j\_{1}}-\\tilde{C} _{i_{2},j\_{2}})\*\\mathfrak{u} _{i_{1},j\_{1},i\_{2},j\_{2}}\\geqslant0.}\\end{array}

\\tilde{S} _{i,j}\\geqslant\\tilde{C}_{i-1,j}.

Eqs.(4)and(5)are two objective functions. Eq.(4)is the maximum
fuzzy completion time of all operations and Eq.(5)is the total machine
workload. Eq.(6)is a formulation to calculate machine  ’s workload.

Eq.(7)describes the relationship between operation O
time and starting time. Eq.(8)ensures operation O
after operation Oi,j−1being finished. Eq.(9)specifies that the successor
operation must wait for the following machine to be idle. Eq.(10)

k
Eq.(7)describes the relationship between operation Oi,j’s completion
time and starting time. Eq.(8)ensures operation Oi,jmust be processed
being finished. Eq.(9)specifies that the successor
operation must wait for the following machine to be idle. Eq.(10)

operation must wait for the following machine to be idle. Eq.(10)
guarantees that job i has only one operation processed on the machine.

guarantees that job i has only one operation processed on the machine.
The range of variables is described in Eqs.(11)and(12).

guarantees that job i has only one operation processed on the machine.
The range of variables is described in Eqs.(11)and(12).

\\mathcal{M}\_{k}

(9)

\\sum\_{k=1}^{m}\\mathbf{x}\_{i,j,k}=1.

3.4. Illustrative example

\\mathbf{x} _{i,j,k}=\\left{\\begin{aligned}{}&{{}1,i f\ O_{i j}\ i s p r o c e s s e d\ o n\\mathcal{M}\_{k},}\ {}&{{}0,o t h e r w i s e.}\ \\end{aligned}\\right.

O\_{i,j-1}

\\alpha,

O\_{i,j}

O\_{2,3}

After computing the TFN of completion time for each operation, the
maximum completion time and total machine workload for the system
will be obtained, and the objective function can be collected.

4. Proposed algorithm

4.1. Framework of RMOEA/D

In this section, framework of the proposed algorithm MOEA/D
based on RL is introduced, which is stated as following: first, the parameters and population are initialized. Then, perform variable neighborhood search for each solution to improve they exploitation. Next, apply
Q-learning to select a parameter T for MOEA/D. Moreover, execute
MOEA/D to generate new solution by using T. Finally, according to
the change of PF’s convergence and diversity, Q-table will be updated.
And the elite archive will store the non-dominated solutions. If the
stopping criterion is not satisfied go to row 4 and continue the iteration.
Algorithm1describes the process of RMOEA/D.

Algorithm 1: RMOEA/D Algorithm.
Input: Population size N, mutation rate R, neighborhood number
vector T, length of parameter memory LP, learning rate 𝛼,
discount factor 𝛾, greedy factor 𝜖, elite archive E
Output: the best solution found so far
1 Initialize the population P sizing N (c.f. Section4.3)
2 Initialize all variables and parameters for algorithm.
3 while the stopping criterion is not satisfied do
4 Perform proposed VNS for each individual. (c.f. Section4.6);
5 Apply Q-learning to allocate the parameter T for MOEA/D(c.f.
Section4.5;
6 Execute MOEA/D algorithm for each individual using parameter
Tiby the discrete crossover and mutation methods. (c.f.
Section4.4);
7 Update population state and Q-table(c.f. Section4.5);
8 Calculate the non-dominated solution set and collect them into
elite archive(c.f. Section4.7);

O\_{3,1},,O\_{2,1},,O\_{1,1},,O\_{2,2}

T\_{i}

\\mathcal{M} _{1},~\\mathcal{M}_{3},~\\mathcal{M}\_{2},

In this paper, two one-dimensional vectors are used to represent
the solution (or agent). The operation sequence is used to indicate the
processing sequence for all operations. And the machine selection is
used to represent the assigned machine for each operation. The two
vectors are set with the same length which is equal to the total number
of operations.
Fig.3displays an encoded solution. The solution representation

Fig.3displays an encoded solution. The solution representation
contains two vectors. The operations sequence is O3,1, O2,1, O1,1, O2,2,
O1,2, O3,2, O1,3, O3,3, O2,3. The machine selection is 1, 3, 2,
2, 2, 3, 2, 2, 1. It is a one-to-one correspondence. For
example, O3,1selected 1and O2,1selected 3. The decoding of a
solution is to assign the appropriate processing time for each operation
on its selected machine according to the operation sequence. When
a solution is decoded, the first vector inFig.3is converted into a
sequence of operations at frist. Then each operation is assigned to a
selected machine from the second vector inFig.3. Finally, the fuzzy
processing times are assigned to the operation. In this paper, each
solution (agent) is decoded into a fuzzy schedule. That is the processing
time is a triangle fuzzy number.

O\_{1,2},,O\_{3,2},,O\_{1,3},,O\_{3,3},,O\_{2,3}

To illustrate the problem,Fig.2presents a Gantt chart of a con-4.3. Initialization strategy
the

In this section, an initial strategy integrated with 3 rules is introduced. Those rules includes Random rule (Gao, Suganthan, Pan, &
Tasgetiren,2015), local minimum processing time (LS) rule (Shaheed
et al.,2018), global minimum workload (GW) rule (Li, Liu, et al.,
2020). LS and GW are aiming at optimizing the objective function. The
descriptions of three strategies are given as follows:

\\mathcal{M}\_{3}

\\mathcal{M} _{2},,\\mathcal{M}_{2},,\\mathcal{M} _{3},,\\mathcal{M}_{2},,\\mathcal{M} _{2},,\\mathcal{M}_{1}

* * *

Fig. 2. A Gantt chart for an example problem.

Fig. 3. Encoding representation.

Random: This rule is simple and ensures the initial population has
high diversity. (1) repeat each job i for 𝛩itimes to generate a scheduling vector. (2) randomly rearrange the sequence for all operations
in the scheduling vector. (3) randomly choose a machine from the
operation’s candidate set for each operation and generate a routing
vector.
LS: This rule aims to reduce the fuzzy makespan (maximum com-

\\varrho\_{i}

LS: This rule aims to reduce the fuzzy makespan (maximum completion time). (1) randomly generate a scheduling vector same as the
Random rule. (2) for each operation, choose the minimum processing
time machine from the candidate set to generate the routing vector.
GW: This rule focuses on lower the total machine workload. (1)

The advantages of the three strategies are combined to obtain a
population of high quality. A method called MIX3 is developed and the
description is given in Algorithm2.

GW: This rule focuses on lower the total machine workload. (1)
put O1,1, O2,1, …,ON,1into scheduling vector and rearrange the sequence. (2) rearrange the sequence of the rest operation and connect
after the former sequence. (3) for each operation, choose an available
machine with the minimum workload. If more than one machine had
the same workload, the machine with minimum processing time would
be selected.

S\_{2}

N e u S\_{2}

Algorithm 2: MIX3 rule.
Input: Population size, Np.
Output: initialization population
1 Perform GW to generate an offspring P1, size ⌊Np∕3⌋;
2 Execute LS to generate an offspring P2, size ⌊Np∕3⌋;
3 Use Random to generate an offspring P3, size ⌊Np∕3⌋;
4 Combine P1, P2 and P3 into Parent size 3× ⌊Np∕3⌋;
5 if size of Parent = Np then
6 terminate the algorithm;
7 else
8 supplement the rest of the solution with Random rule;

\\mathbb{P}\_{1},

O\_{1,1},,O\_{2,1},\\ldots,O\_{N,1}

N e u S\_{1}

\\mathbb{P}\_{2}.

S\_{1}

\\lfloor p/3\\rfloor

\\mathbb{P} _{1},,\\mathbb{P}_{2}

N e u S\_{1}.

M\_{1}

\\mathbb{P}\_{3}

\\mathbb{P}\_{3}.

\\lfloor p/3\\rfloor;

4.4. Crossover and mutation

To get a large exploring step, precedence operation crossover (POX)
and universal crossover (UX) (Gao, Suganthan, Chua, et al.,2015) are
adopted for MOFFJSP are applied.Fig.4gives two examples. The
description is given as follows:
POX for operation sequence: (1) Randomly divide job set into two

3\\times\\lfloor N p/3\\rfloor

N e u S\_{2}

{boldsymbol\\varsigma}=N

POX for operation sequence: (1) Randomly divide job set into two
subset J1and J2. (2) Select two solutions S1and S2. For each job
belonging to J1, copy their operations into NewS1. And for each job
belonging to J , copy their operations into NewS . (3) There exists too2 2
many space that is not filled with operation in NewS1and NewS .2
From the part of S , copy the operation which does2not appear in
NewS1to the vacant positions in NewS1from left to right according
to the order of the sequence in S2. And do the similar thing to NewS2.
The procedure is illustrated inFig.4.
UX for machine selection: (1) Randomly generate a 0-1 vector which

\\mathbb{J}\_{1}

N e u S\_{1}

\\mathbb{A}\_{2}

UX for machine selection: (1) Randomly generate a 0-1 vector which
length equals the total number of operations. (2) Select two solution M1
and M2. Exchange the value in M1and M2, where it is 1 at the same
position in 0-1 vector. The procedure is illustrated inFig.4.
Operation sequence mutation: randomly select two positions in

N e u S\_{1}

M\_{2}

R\_{t+1}

M\_{1}

position in 0-1 vector. The procedure is illustrated inFig.4.
Operation sequence mutation: randomly select two positions in
the operation sequence and exchange the value. Machine selection
mutation: randomly select two positions in the machine selection and
choose a new machine from its candidate set.

4.5. Parameter adaption strategy based on Q-learning

4.5.1. Brief introduction of Q-learning
Q-learning was first proposed in 1992

Q(S\_{t},A\_{t})=Q(S\_{t},A\_{t})+\\alpha\[R\_{t}+\\gamma\\operatorname\*{m a x}(Q(S\_{t+1},A\_{t})-Q(S\_{t},A\_{t}))\]

Q-learning was first proposed in 1992 byWatkins and Dayan
(1992), and it has been one of the most well-known RL algorithms.
Q-learning consists of a quintuple (A, E,C,S,R), which means Agents,
Environment, Action set, State set and Reward. As shown inFig.5, the
agent bases on its state Stat time t in the environment and executes
an action At. Then agent will get a reward Rt+1and its state will turn
into a new state St+1. The Q value is updated based on the following
formula:

S\_{t+1}

S\_{t}

A\_{t}

* * *

(a) An example of POX

the differential value between realistic Q value and estimated value,
agent can adjust the gap between reality. 𝛼 is the learning rate, which
is between 0 and 1. 𝛾 is the discount factor is also in \[0, 1\]. When 𝛾 is
close to 1, Q value is more affected by future state. On the contrary,
the more 𝛾 is close to 0, Q value is more focus on the current state. It
is a Markov decision process. The current state St+1can affect the later
state St+i. Rtis the reward after executing action Ai.

Fig. 5. The agent–environment interaction of RL.

Fig. 4. An example of UX.

\\bar{d}

(b) Example of machine selection crossover

S\_{t+1}

4.5.2. Agent and action definition
In the MOEA, the Pareto Front (PF) is the best solution set calculated

S\_{t+i},\ R R\_{t}

\\varDelta C V>0

In the MOEA, the Pareto Front (PF) is the best solution set calculated
by MOEA and it reflects the algorithm’s ability. By means of Q-learning,
the population will be guided to choose the best parameter T to
improve PF’s diversity. Increasing the diversity of the whole population
can improve the diversity of the PF. So the PF in each generation is
abstracted as an agent to reflect the successful parameter selection. The
four candidate values T = 5, 10, 15, 20 are defined as actions.

C V(P,P^{ _})=\\frac{\\sqrt{\\sum\_{y\\in P}\\operatorname_{m i n}\_{x\\in P^{\*}}d i s(x,y)^{2}}}{\|P\|}

(16)

P^{\*}

\\varDelta C V=C V\_{i-1}-C V\_{i}

\\varDelta D V=D V\_{i}-C V\_{i-1}

point because the true PF of the FFJSP instance cannot be calculated.
The smaller CV is the better convergence is and CV > 0. In Eq.(15),
diis the Euclid distance of two adjacent points in PF. d̄ is the average
value of di. The bigger DV is the better diversity is and DV > 0.
During the evolutionary, the CV and DV of PF have four com-

D V=\\frac{\\sum\_{i=1}^{N-1}\|d\_{i}-\\bar{d}\|}{(N-1)\\bar{d}}

value of di. The bigger DV is the better diversity is and DV > 0.
During the evolutionary, the CV and DV of PF have four combination conditions. (1) CV > 0 and DV > 0; (2) CV > 0 and
DV ⩽ 0; (3) CV ⩽ 0 and DV > 0; (4) CV ⩽ 0 and DV ⩽ 0;
(1) and (2) occurs during the period of evolutionary. (3) and (4) will
happen when the population has converged. These four conditions are
regarded as four states of agent.

(18)

d\_{i}

\\varDelta C V,>,0

\\vartriangle C V\\leqslant0

\ \\varDelta V V,>,0;

d\_{i}

\ \\varDelta V V;\\leqslant;0;

After executing an action, the agent will get a reward, which might
be positive or negative. The reward is defined as below:
{
10,DV > 0,

S\_{t+1}

R e w a r d=\\left{\\begin{array}{l l}{10,\\varDelta D V>0,}\ {0,\\varDelta D V\\leqslant0.}\\end{array}\\right.

If the diversity of PF is better the chosen action (parameter) will get
a reward and update the Q-table. On the contrary, the reward will be
zero. It is worth noting that the reward value used most is 10 from the
previous studies. We also used to consider a penalty when the agent
fails to increase DV. However, this penalty will decrease the impact of
successful reward. Thus, to let the positive reward more significant, the
penalty is not considered in this work.
4.5.5. Q-learning for parameter self-adaption

4.5.5. Q-learning for parameter self-adaption
Based on the illustration above, a parameter adaption strategy based
on Q-learning namely Q-PAS is designed. Algorithm3described the
details of Q-PAS. First, a Q-table is initialed to 0. Second, the agent’s
state Stshould be got. Next, 𝜖-greedy selection strategy is applied to
permit each action to have the probability to be selected. It is worth
mentioning that the 𝜖-greedy strategy is opposite to the traditional
strategy, for the convenience of setting parameters in the experiment.
Then, action Atis executed to MOEA/D and PF  is got. Moreover,
CV and DV are updated. The next state of agent St+1is got. Finally,
Q(S ,A ) in Q-table is updated by Eq.(15).t t

4.6. Variable neighborhood search based on RL

Q(S\_{t},A\_{t})

* * *

Algorithm 3: Parameter adaption strategy based on Q-learning Algorithm 4: Variable neighborhood search based on RL
(Q-PAS). (RVNS).
Input: Population P, greedy factor 𝜖, learning rate 𝛼, discount factor Input: solution P(i),  = {1, 2,..., 5} selection probability of each
𝛾. local search, reference point Z∗, weight vector 𝜆i
′
Output: Q\_table Output: new solution P(i)
1 Q\_table (4,4) ← 0; CV ← 1; DV ← 1; 1 Execute Roulette Algorithm according to  to assign a local search
CVi−1= CVi= DVi−1= DVi= 0; LSito P(i);
′
2 while the stopping criterion is satisfied do 2 Adopt LSito get a new solution P(i);
te′i ∗ te i ∗
3 CVi−1= CVi; DVi−1= DVi; 3 if g (P(i) \|𝜆,Z) < g (P(i)\|𝜆,Z) then
′
4 Calculate agent’s state St(c.f. Section4.5.3) 4 P(i) = P(i), nsi= nsi+1
5 if rand < 𝜖 then5 else
6 Select the max Q(St,Ai) action At, i = 1, 2, 3, 46 nf = nf +1
i i
7 else7 Count the ns and nf i = 1,...,n. Save this record into the back of
i i
8 Randomly select a action Atsuccess memory and failure memory;
9 Execute action Atfor MOEA/D to update P and Get PF  8 if length of SM > LP then
10 Calculate CViand DViby equation (16) and (17) 9 Delete the first record in SM and FM;
11 CV = CVi−1− CVi, DV = DVi− DVi−1 10 for i = 1 to n∑do
LP∑LP
12 Get action At’s reward R(St,At) by equation (20) 11 SR(i) =j=1nsi,j; FR(i) =j=1nfi,j;
13 Calculate the new solution x ’s state S (c.f. Section4.5.3)SR(i)
new t+1 12  (i) =;
SR(i)+FR(i)
14 Q(St,At) = Q(St,At)+ 𝛼\[R(St,At)+ 𝛾 max(Q(St+1,At)− Q(St,At))\]
13 for i = 1 to n do
 (i) (i)
14  (i) = ∑n
i=1
LS2: Randomly select an operation Oi,jand move it to another

\\mathbb{P},

\\alpha,

\\gamma,

(4,\ )\\leftarrow0;,\\varDelta V V\\leftarrow1;\\varDelta D V\\leftarrow1;

C V\_{i-1}=C V\_{i}=D V\_{i-1}=D V\_{i}=0;

C V\_{i-1}=C V\_{i};,D V\_{i-1}=D V\_{i};

Q(S\_{t},A\_{i})

A\_{t},,i=1,2,3,4

A\_{t}

C V\_{i}

\\notDelta C V=C!{V} _{i-1}-C!{V_{i}},,\\Delta D V=D!{V\_{i}}-D!{V\_{i-1}}

R(S\_{t},A\_{t})

A}\_{t}{}^{\\prime}

x\_{n e w}\\mathbf s{}

S\_{t+}

LS2: Randomly select an operation Oi,jand move it to another
machine with minimum processing time.
LS3: Find the maximum workload machine . Randomly select an

LS3: Find the maximum workload machine . Randomly select an
operation which processed by  and move it to another machine ′.
LS4: Randomly choose two positions on the operation sequence and

Q(S\_{t},A\_{t})=Q(S\_{t},A\_{t})+\\alpha\[R(S\_{t},A\_{t})+\\gamma\\operatorname\*{m a x}(Q(S\_{t+1},A\_{t})-Q(S\_{t},A\_{t}))\]

LS4: Randomly choose two positions on the operation sequence and
exchange the value.
LS5: Randomly choose two positions on the operation sequence and

RVNS: (1) Execute roulette algorithm to assign a local search LSito
each solution P(i). (2) Adopt local search LSito generate a new solution
P′and judge whether the old solution can be updated. (3) Count the
number of updating old solution successfully np and unsuccessfully nf
as shown inFig.6. Save the record and insert it into the tail of success
memory SM and failure memory FM. (4) If the length of SM and
FM is longer than LP, delete the first record in SM and FM. Then
update the probability of each parameter by sum each column in SM
and FM as SR and FR and use SR to divide the sum of SR and FR.
(5) Normalize the probability of each parameter  (i) to ensure the sum
of probabilities equal to 1.

\\mathcal{M}^{\\prime}

\\mathbb{P}(i)

Z^{\*},

N p,

\\overline{{\\mathbb{P}(i),,\\mathcal{P}={\\mathcal{P} _{1},\\mathcal{P}_{2},...,\\mathcal{P}\_{5}}}}

\\mathbb{P}(i)^{\\prime}

p

(RVNS).
Input: solution P(i),  = {1, 2,..., 5} selection probability of each
local search, reference point Z∗, weight vector 𝜆i
′
Output: new solution P(i)
1 Execute Roulette Algorithm according to  to assign a local search
LSito P(i);
′
2 Adopt LSito get a new solution P(i);
te′i ∗ te i ∗
3 if g (P(i) \|𝜆,Z) < g (P(i)\|𝜆,Z) then
′
4 P(i) = P(i), nsi= nsi+1
5 else
6 nfi= nfi+1
7 Count the nsiand nfii = 1,...,n. Save this record into the back of
success memory and failure memory;
8 if length of SM > LP then
9 Delete the first record in SM and FM;
10 for i = 1 to n∑do
LP∑LP
11 SR(i) =j=1nsi,j; FR(i) =j=1nfi,j;
SR(i)
12  (i) =;
SR(i)+FR(i)
13 for i = 1 to n do
 (i) (i)
14  (i) = ∑n
i=1

\\mathbb{P}(i)^{\\prime}

g^{l e}(\\mathbb{P}(i)^{\\prime}\|\\lambda^{i},Z^{ _})<g^{l e}(\\mathbb{P}(i)\|\\lambda^{i},Z^{_})

L!{S}\_{i}

L!{S}\_{i}

\\box{\\begin{array}{r}{\\mathbb{P}(i)=\\mathbb{P}(i)^{'},,n s\_{i}=n s\_{i}+1}\ }\\end{array}

F M;

{mathtt{n s}}\_{\\mathsf{n},G mathsf{-L L P}}

\\mathtt{n s}\_{\\mathsf{n},\\widehat{G}\ \ {l},mathsf l,mathsf}{\ {ll}}

n f\_{i}\ i=1,...,n.

\ \ \ \ n f\_{i}=n f\_{i}+1

\\begin{array}{r}{\\mathcal{P}(i)=\\frac{S R(i)}{S R(i)+F R(i)};}\\end{array}

\\delta M>L P

\\mathbf{n f}\_{\\mathsf{n},G-L P}

\\begin{array}{r}{S R(i)=\\sum\_{j=1}^{L P}n s\_{i,j};,F R(i)=\\sum\_{j=1}^{L P}n f\_{i,j};}\\end{array}

{\\bf n s}\_{2,G-L P}

{\\mathtt n s}\_{2,G\ L P+1}

\\mathtt{n s}\_{\\mathsf{n},G\\cdot P+1}

\\begin{array}{r}{\\mathcal{P}(i)=\\frac{p\_i({}}}{{{sumsum\_{i=1}^{n}{\\mathcal{P}}(i)}}}\\end{array}

T\_{i})

\\mathbf{n f}\_{2,G-L P}

{bf}s s\_{1,G-1}

\\mathbf{n f\_{l1,G L P}}

\\mathbf{n f}\_{\\mathsf{n},G\\cdot1}

\\mathbf{n f}\_{2,G-L P+1}

1.On the MIX3, the complexity is (Np), where Np is the size of
the population.
2.On the MOEA/D, the complexity is (Iter ∗ Np ∗ 2 ∗ T),

Fig. 6. Success memory and failure memory.

2.On the MOEA/D, the complexity is (Iter ∗ Np ∗ 2 ∗ Ti),
where Iter is the number of iteration and Tiis the number of
the neighborhood.
3.On the Q-PAS, the complexity is (Iter). where LP is the length

| Algorithm 5: Update Elite Archive |  |
| --- | --- |
| Input: Archive A, population size Np, PF of this generation PFOutput: Archive A |  |
| 1 | A←A∪PF; |
| 2 | if length of A＞Np then |
| 3 | A←Non-dominated-sort(A) |

N p\_

3.On the Q-PAS, the complexity is (Iter). where LP is the length
of success memory and failure memory and T is the vector of
candidate parameter.

\\mathcal{A}\\leftarrow\\mathcal{A}\\cup\\mathcal{P F};

* * *

Table 2
Scale illustration of instances.

| Name | N | M | SH | C-FFJSP | Literature |
| --- | --- | --- | --- | --- | --- |
| D1 | 10 | 10 | 40 |  |  |
| D2 | 10 | 10 | 40 |  |  |
| D3 | 10 | 10 | 50 |  |  |
| D4 | 10 | 10 | 50 | Y | (Lei, 2010) |
| D5 | 15 | 10 | 80 | Y | Lei(2012) |
| R1 | 5 | 4 | 23 |  |  |
| R2 | 8 | 8 | 64 |  |  |
| R3 | 10 | 6 | 81 |  |  |
| R4 | 10 | 10 | 100 |  |  |
| R5 | 15 | 8 | 171 |  |  |
| R6 | 15 | 10 | 185 |  |  |
| R7 | 20 | 10 | 308 |  | (Gao,Suganthan,Pan,&Tasgetiren,2015) |
| R8 | 20 | 15 | 355 |  |  |
| FMk1 | 10 | 6 | 55 |  |  |
| FMk2 | 10 | 6 | 58 |  |  |
| FMk3 | 15 | 8 | 150 |  |  |
| FMk4 | 15 | 8 | 90 |  |  |
| FMk5 | 15 | 4 | 106 |  |  |
| FMk6 | 10 | 15 | 150 | N |  |
| FMk7 | 20 | 5 | 100 |  |  |
| FMk8 | 20 | 10 | 225 |  |  |
| FMk9 | 20 | 10 | 240 | (Brandimarte,1993) |  |
| FMk10 | 20 | 15 | 240 |  |  |

Table 3
Q-table after 200 iterations.

T=5

|  | T=5 | T=10 | T=15 | T=20 |
| --- | --- | --- | --- | --- |
| State1(ΔCV>0, ΔDV>0) | 0 | 1.798693 | 0 | 6.743954 |
| State2(ΔCV>0, ΔDV≤0) | 0 | 0.323268 | 0 | 5.271637 |
| State3(ΔCV≤0, ΔDV>0) | 4.539988 | 0.766909 | 0 | 0 |
| State4(ΔCV≤0, ΔDV≤0) | 2.847096 | 2.28377 | 2.911486 | 1.152641 |

T=10

T=15

T=20

(\\Delta C V>0,\\Delta D V>0)

(\\Delta C V>0,\\Delta D V\\leqslant0)

(\\Delta C V\\leqslant0,\\Delta D V>0)

(\\Delta C V\\leqslant0,\\Delta D V\\leqslant0)

Table 4
Average rankings of the variant algorithms
(Friedman) with p-value = 0.00015.
Algorithm Ranking

(19)

| Algorithm | Ranking |
| --- | --- |
| RMOEA/D1 | 4.6087 |
| RMOEA/D2 | 4.3913 |
| RMOEA/D3 | 3.6087 |
| RMOEA/D4 | 3.087 |
| RMOEA/D5 | 2.913 |
| RMOEA/D | 2.3913 |

4.On proposed the RVNS, the complexity is (Iter ∗ 5 ∗ LP).
5.On proposed the Elite archive, the complexity is (Iter ∗ Np2).

4.On proposed the RVNS, the complexity is (Iter ∗ 5 ∗ LP).
5.On proposed the Elite archive, the complexity is (Iter ∗ Np2).
Therefore, the complexity of RMOEA/D is (Iter ∗ Np2)

In Section4, the RMOEA/D algorithm has been described in details.
In this section, we design detailed experiments to evaluate RMOEA/D’s
performance. RMOEA/D and the comparison algorithms are coded in
MATLAB on an Intel Core i7 6700 CPU @ 3.4 GHz with 8G RAM. For
fairness, all algorithm runs 30 independent times on each instance.
Noting that, to verify the convergence and diversity of the proposed
algorithm, after 30 independent runs, the average results are collected
for performance comparison.
The comparison algorithms are well-known multi-objective opti-

\\mathcal{O}(I t e r _5_ L P).

5. Experimental results

Main Effects Plot for HV metric

Fig. 7. Main effects plot of HV metric.

value as the performance measures. HV is calculated as follows:

where P is the PF calculated by algorithms, r is the reference point of
all PFs, r = (1, 1). x is the non-domination solution in PF and x needs to
f1 −f1min f2 −f2min
be normalized, which is x = (,). v is the volume of
f1max −f1min f2max −f2min
the hypercube which is bounded by x and r. HV unions all hypercubes
for each non-domination solution x to generate a irregular hypercube
and calculate its volume. It is worth mentioning that all x should be
normalized in same reference point. The bigger HV is the convergence
and diversity of algorithm is better.

H V(P,r)=\\bigcup\_{\\mathbf{x}\\in P}^{P}v(\\mathbf{x},r).

\\mathbf{x}=(\\frac{f\_{1}-f\_{1m i n}}{f\_{1m a x}-f\_{1m i n}},\\frac{f\_{2}-f\_{2m i n}}{f\_{2m a x}-f\_{2m i n}})

After introducing the experiment design, the experiment instances
are reported. Three benchmarks are selected to verify the convergence
and diversity of the proposed RMOEA/D. The first benchmark Lei01
and Lei02 are obtained from (Lei,2010,2012). The second benchmark
Remanu is provided by (Gao, Suganthan, Pan, & Tasgetiren,2015).
All the instances are FFJSP with fuzzy processing time. Moreover,
we transformed a flexible job shop scheduling problem benchmark
Mk (Brandimarte,1993). Refer to each processing time b, two integers
a and c were randomly generated in interval \[0, b∕2\]. (a,b,c) is a TFN.
Therefore, Mk benchmark is converted to the FFJSP benchmark FMk.
As shown inTable2, the scale of all instances can be obtained.

5.1. Experimental instances

\\cdot,\\gamma=0.9,,0.8,,0.7,,0.6

5.2. Experimental parameters

\\cdot\\alpha=0.1,0.2,0.3,0.4.

\\cdot\ varepsilon0==0.95,0.90,0.85,0.80.

An orthogonal array L (44) is adopted in this calibration experi-
16
ment. For fairness, each parameter runs 30 independent times. According to our previous work, the population size Np = 100 is the best.
The other parameters include the max generation G = 200 and the
neighborhood number vector T in Section4.5which has 4 candidate
parameters T = {5, 10, 15, 20}. We collect the average HV value for

T;=;{5,10,15,20}

* * *

Table 5
Comparison of HV result for variants of RMOEA/D.

| Instances | RMOEA/D1 | RMOEA/D2 | RMOEA/D3 | RMOEA/D4 | RMOEA/D5 | RMOEA/D |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | 0.096358 | 0.097446 | 0.099396 | 0.098672 | 0.099119 | 0.099671 |
| D2 | 0.101298 | 0.101899 | 0.103487 | 0.103056 | 0.103296 | 0.103148 |
| D3 | 0.064942 | 0.067186 | 0.066809 | 0.067699 | 0.067898 | 0.06737 |
| D4 | 0.057389 | 0.058733 | 0.058901 | 0.059791 | 0.058731 | 0.05847 |
| D5 | 0.048071 | 0.050738 | 0.052962 | 0.053041 | 0.052376 | 0.052245 |
| R1 | 0.050917 | 0.050711 | 0.050713 | 0.050458 | 0.050879 | 0.050919 |
| R2 | 0.031251 | 0.032267 | 0.033204 | 0.031718 | 0.031889 | 0.031856 |
| R3 | 0.034702 | 0.035149 | 0.035013 | 0.03601 | 0.036115 | 0.035549 |
| R4 | 0.039239 | 0.039681 | 0.041548 | 0.041979 | 0.041942 | 0.042023 |
| R5 | 0.042622 | 0.044959 | 0.045299 | 0.045912 | 0.046284 | 0.046105 |
| R6 | 0.045522 | 0.050659 | 0.051547 | 0.052755 | 0.051897 | 0.051946 |
| R7 | 0.041646 | 0.054828 | 0.056424 | 0.057074 | 0.057371 | 0.057465 |
| R8 | 0.043417 | 0.069881 | 0.073654 | 0.073011 | 0.073377 | 0.074284 |
| FMk01 | 0.058504 | 0.055246 | 0.056479 | 0.057164 | 0.056871 | 0.057207 |
| FMk02 | 0.042559 | 0.040818 | 0.042238 | 0.043146 | 0.042341 | 0.042327 |
| FMk03 | 0.06336 | 0.063915 | 0.064411 | 0.063918 | 0.064357 | 0.064147 |
| FMk04 | 0.094665 | 0.095224 | 0.094696 | 0.095569 | 0.095808 | 0.09624 |
| FMk05 | 0.048044 | 0.048418 | 0.048343 | 0.048325 | 0.048403 | 0.048468 |
| FMk06 | 0.045506 | 0.043584 | 0.044403 | 0.045113 | 0.044132 | 0.044598 |
| FMk07 | 0.071437 | 0.070497 | 0.070504 | 0.070188 | 0.070395 | 0.070645 |
| FMk08 | 0.022032 | 0.021521 | 0.021134 | 0.021609 | 0.021323 | 0.021721 |
| FMk09 | 0.042823 | 0.043149 | 0.042166 | 0.04264 | 0.042568 | 0.042508 |

This section is to prove the effectiveness of Q-learning by picture
and Q-table analysis.Fig.8shows the tendency chart of CV and DV of
a one-time run. This example is a solution extract from instance D1. As
shown inFig.8, each trough to peak can be regarded as a cycle. When
population start to evolution, it converge fast because of local search,
crossover, and mutation strategies. And DV reduces quickly. After
exploring a period of time, solutions fall into local optima and Q-PAS
guides population selects the best parameter to increase the diversity.
So the DV increase fast. When next cycle starts, the regular for DV
is obvious which reduces first and increase then. That can prove that
Q-PAS has a great impact for increasing diversity when convergence is
hard to improve.Table3gives the Q-table after 200 iterations. In the
early period of iteration, CV reduce fast and T needs to be set bigger
such as 20 to improve the DV. When solutions fall into local optima
and CV is hard to increase, T need to be adjusted smaller such as 5
to increase DV. If both CV and DV seldom increase, then T = 5 or

30 runs.Fig.7shows the main effects plot of four parameters for HV
metric. The higher the metric values is, the better the performance
is. Based on the comprehensive observation, the best configuration of
parametric value is set as LP = 40, 𝛾 = 0.6, 𝛼 = 0.4, and 𝜖 = 0.8.

5.3. Discussion about Q-PAS

\\epsilon=0.8.

10 may help the DV increase. From what has been discussed above,
Q-PAS is an efficient strategy to let population dynamically select the
best parameter in every state of agent.

5.4. Effectiveness of each improvement part of RMOEA/D

After proving the effectiveness of Q-learning the other improvement
parts are verified to be efficiency in this section. We set several variants of RMOEA/D by adding the improvement part to MOEA/D step
by step. RMOEA/D1 is the pure MOEA/D without any improvement,
RMOEA/D2 is the MOEA/D with initial strategy, RMOEA/D3 is the
RMOEA/D2 with randomly selection VNS, RMOEA/D4 is RMOEA/D3
with Q-PAS, RMOEA/D5 is RMOEA/D4 with Elite archive, RMOEA/D
change VNS in RMOEA/D5 to RVNS. For fairness, all algorithms run
30 independent times.Table5shows the HV metric result of the comparison. Bold and gray value means the best result for each instance.
And Friedman rank is shown inTable4, each part improves the result
against the last one. That proves the effectiveness of each part.

\\mathtt{P-v a l u e}=1.312,\*,10^{-17}<0.05.

5.5. Comparison and discussion

inTable6.
Table7shows the HV results for algorithms comparison. The bold
and gray value is the best average HV value in this instance. RMOEA/D
is superior to other algorithms. AndTable8shows the Friedman
ranking of all algorithms. It can be concluded that RMOEA/D is the
best algorithm on all instances and P-value = 1.312 ∗ 10−17< 0.05.
Table9shows the result of the Wilcoxon test. The results show that
RMOEA/D is significantly better than other algorithms and all p-value
is smaller than 0.05.

* * *

Table 6
Parameters setting.

Fig. 10. Comparison results for metric convergence.

| Algorithm | Special parameters | Common parameters |
| --- | --- | --- |
| RMOEA/D | memory size LP=40, learning rare $\\alpha=0.4$, discount factor $\\gamma=0.6$, greedy factor $\\epsilon=0.8$, neighborhood vector $T=\[5,10,15,20\]$ | popsize NP=100, weight vectors number $\\lambda=100$, mutation rate R=0.8 |
| MOEA/D | number of neighborhood T=10 |  |
| MOEA/D-M2M | number of neighborhood T=10, subpop size S=10 |  |
| NSGA-III | number of neighborhood T=10 |  |
| NSGA-II | - | R=0.8,Np=100 |
| IAIS | clone-popsize nc=10, popsize NP=nc\*(nc+1)/2,crowding degree threshold CRmax=1\*10^{-4},temperature rate w=0.5,initial temperature $T\_{0}=1\\times10^{4}$ |  |

N P=100,

\\lambda=100,

The proposed RMOEA/D is better than other compared algorithms.
There are four reasons below. (1) The initial strategy mentioned in
Section4.3generated an initial
and diversity, which results in
The parameter adaption strategy
Section4.5guides the algorithm chooses the most suitable parameter T

The proposed RMOEA/D is better than other compared algorithms.
There are four reasons below. (1) The initial strategy mentioned in
initial population with high convergence
in the algorithm converging faster. (2)
strategy based on Q-learning proposed in
Section4.5guides the algorithm chooses the most suitable parameter T

Section4.5guides the algorithm chooses the most suitable parameter T
to increasing the diversity of PF. (3) RMOEA/D adopted VNS based on
RL described in Section4.6improves the convergence of the algorithm,
so the results have better convergence. (4) The Elite archive collects

T=10,

5.5.2. Discussion
The proposed RMOEA/D is better than other compared algorithms.

n n=c10,

R=0.8,;N p=100

N P=n\_{c}\*(n\_{c}+1)/2,

C R m a x=1\*10^{-4},

T\_{0}=1\*10^{4}

Fig. 11. The Gantt chart for solution with best makespan.

This paper proposed a RMOEA/D that combined two RL techniques
to solve multi-objective fuzzy flexible job shop scheduling problems.
The objective is to minimize the fuzzy makespan and total workload.
As a classical adaption technique, RL can guide the algorithm to
automatically select the best parameter or local search strategy. A
novel Q-learning parameter adaption method including state definition,
action definition, and reward definition is designed to help multiobjective optimization algorithm MOEA/D automatically select the
parameter. Nevertheless, a RL method based on two historical memories is adopted to make the algorithm choose the local search strategy
with the highest success probability. Next, to improve the usage rate of
the historical solutions, an elite archive is used to collect the elite solution during iteration. Moreover, an initial strategy adopting three initial
rules is applied to get an initial population with high convergence

the abandoned solution with high quality during historical iteration.
This supplemented the final PF and lead the RMOEA/D to have better
results.

6. Conclusion

* * *

Table 7
HV result for comparison with other algorithms.

| Instances | IAIS | MOEA/D | MOEA/D-M2M | NSGA-II | NSGA-III | RMOEA/D |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | 0.101648 | 0.121501 | 0.102406 | 0.112091 | 0.10712 | 0.125066 |
| D2 | 0.126441 | 0.145188 | 0.133167 | 0.135597 | 0.130158 | 0.147033 |
| D3 | 0.107201 | 0.113763 | 0.096045 | 0.10405 | 0.099619 | 0.117255 |
| D4 | 0.106326 | 0.111984 | 0.090837 | 0.103181 | 0.096756 | 0.113831 |
| D5 | 0.14925 | 0.153384 | 0.11185 | 0.140018 | 0.129151 | 0.158382 |
| R1 | 0.115783 | 0.123475 | 0.113377 | 0.121941 | 0.122881 | 0.123476 |
| R2 | 0.067973 | 0.073794 | 0.061281 | 0.071102 | 0.068841 | 0.074325 |
| R3 | 0.089022 | 0.093673 | 0.092318 | 0.091856 | 0.089554 | 0.09424 |
| R4 | 0.096069 | 0.103368 | 0.082595 | 0.098912 | 0.093185 | 0.105898 |
| R5 | 0.102766 | 0.107348 | 0.080571 | 0.098157 | 0.085406 | 0.111737 |
| R6 | 0.135977 | 0.135997 | 0.07776 | 0.123481 | 0.105439 | 0.144087 |
| R7 | 0.17597 | 0.170876 | 0.064412 | 0.150029 | 0.11291 | 0.19205 |
| R8 | 0.193543 | 0.165819 | 0.068542 | 0.125642 | 0.083388 | 0.206107 |
| FMk01 | 0.060938 | 0.075054 | 0.067977 | 0.070224 | 0.06702 | 0.073803 |
| FMk02 | 0.058621 | 0.070369 | 0.062152 | 0.066106 | 0.064245 | 0.070362 |
| FMk03 | 0.056997 | 0.078065 | 0.067212 | 0.07017 | 0.068431 | 0.07847 |
| FMk04 | 0.068268 | 0.097126 | 0.082704 | 0.08333 | 0.077104 | 0.098794 |
| FMk05 | 0.054107 | 0.06207 | 0.056213 | 0.056679 | 0.055799 | 0.062563 |
| FMk06 | 0.096688 | 0.13492 | 0.098165 | 0.131115 | 0.116562 | 0.132173 |
| FMk07 | 0.084151 | 0.099128 | 0.09029 | 0.091868 | 0.089918 | 0.099004 |
| FMk08 | 0.038212 | 0.043297 | 0.036438 | 0.038692 | 0.038112 | 0.04306 |
| FMk09 | 0.082438 | 0.092606 | 0.071231 | 0.089952 | 0.083816 | 0.09258 |

Table 8
Average

| Average rankings of the comparison algorithms (Friedman) with p-value=1.31285e-17. |
| --- |

| Algorithm | Ranking |
| --- | --- |
| MOEA/D-M2M | 5.2609 |
| NSGA-III | 4.6522 |
| IAIS | 4.6087 |
| NSGA-II | 3.3913 |
| MOEA/D | 1.8261 |
| RMOEA/D | 1.2609 |

Table 9
Results obtained by the Wilcoxon test for algorithm RMOEA/D.

Table 9
Results obtained by the Wilcoxon test for algorithm RMOEA/D.
VS R+R−Exact p-value

Fig. 12. The Gantt chart for solution with best total workload.

R^{+}

R^{-}

algorithm is superior to others. In conclusion, the RMEOA/D is suitable
to solve multi-objective fuzzy flexible job shop scheduling problems
under a high level of uncertainty. And RL technique is an important
method to realize parameter and strategy self-adaption.
Considering future research, first, it is interesting to increase the

Considering future research, first, it is interesting to increase the
objective by more than three and applied RMOEA/D to solve the super
multi-objective problem. Second, other types problem such as distribute
flow job shop problem with fuzzy processing time is also of interest.
Third, applied more complex techniques such as deep Q-network is
another research direction.

Ahmadi, E., Goldengorin, B., Süer, G. A., & Mosadegh, H. (2018). A hybrid method of
2-TSP and novel learning-based GA for job sequencing and tool switching problem.
Applied Soft Computing, 65, 214–229.
Brandimarte, P. (1993). Routing and scheduling in a flexible job shop by tabu search.
Annals of Operations Research, 41(3), 157–183.
Caldeira, R. H., & Gnanavelbabu, A. (2021). A Pareto based discrete jaya algorithm
for multi-objective flexible job shop scheduling problem. Expert Systems with
Applications, 170, Article 114567.

CRediT authorship contribution statement

Rui Li: Resources, Project administration, Software, Data curation, Writing – original draft, Writing – review & editing. Wenyin
Gong: Funding acquisition, Supervision, Conceptualization, Methodology, Writing – review & editing. Chao Lu: Methodology, Writing –
review & editing.

The dataset and code can be downloaded from [https://cuglirui](https://cuglirui/).
github.io/downloads.htm

Acknowledgments

* * *

Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization algorithm
using reference-point-based nondominated sorting approach, Part I: Solving problems with box constraints. IEEE Transactions on Evolutionary Computation, 18(4),
577–601.
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective

Dorfeshan, Y., Tavakkoli-Moghaddam, R., Mousavi, S. M., & Vahedi-Nouri, B. (2020). A
new weighted distance-based approximation methodology for flow shop scheduling
group decisions under the interval-valued fuzzy processing time. Applied Soft
Computing, 91, Article 106248.
Du, Y., Xing, L., Zhang, J., Chen, Y., & He, Y. (2019). MOEA based memetic algorithms

Gao, K. Z., Suganthan, P. N., Chua, T. J., Chong, C. S., Cai, T. X., & Pan, Q. K.
(2015). A two-stage artificial bee colony algorithm scheduling flexible job-shop
scheduling problem with new job insertion. Expert Systems with Applications, 42(21),
7652–7663.
Gao, K. Z., Suganthan, P. N., Pan, Q. K., Chua, T. J., Chong, C. S., & Cai, T. X.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective
genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2),
182–197.
Dorfeshan, Y., Tavakkoli-Moghaddam, R., Mousavi, S. M., & Vahedi-Nouri, B. (2020). A

Han, B. A., & Yang, J. J. (2020). Research on adaptive job shop scheduling problems
based on dueling double DQN. IEEE Access, 8, 186474–186495.
Lang, S., Reggelin, T., Schmidt, J., Müller, M., & Nahhas, A. (2021). NeuroEvolution of

Gao, K. Z., Suganthan, P. N., Pan, Q. K., Chua, T. J., Chong, C. S., & Cai, T. X.
(2016). An improved artificial bee colony algorithm for flexible job-shop scheduling
problem with fuzzy processing time. Expert Systems with Applications, 65, 52–67.
Gao, K. Z., Suganthan, P. N., Pan, Q. K., & Tasgetiren, M. F. (2015). An effective discrete

Gao, K. Z., Suganthan, P. N., Pan, Q. K., & Tasgetiren, M. F. (2015). An effective discrete
harmony search algorithm for flexible job shop scheduling problem with fuzzy
processing time. International Journal of Productions Research, 53(19), 5896–5911.
Gong, W., Liao, Z., Mi, X., Wang, L., & Guo, Y. (2021). Nonlinear equations solving

Lang, S., Reggelin, T., Schmidt, J., Müller, M., & Nahhas, A. (2021). NeuroEvolution of
augmenting topologies for solving a two-stage hybrid flow shop scheduling problem: A comparison of different solution strategies. Expert Systems with Applications,
172, Article 114666.
Lei, D. (2010). A genetic algorithm for flexible job shop scheduling with fuzzy

processing time. International Journal of Productions Research, 53(19), 5896–5911.
Gong, W., Liao, Z., Mi, X., Wang, L., & Guo, Y. (2021). Nonlinear equations solving
with intelligent optimization algorithms: A survey. Complex System Modeling
Simulation, 1(1), 15–32.
Han, B. A., & Yang, J. J. (2020). Research on adaptive job shop scheduling problems

Lei, D. (2010). A genetic algorithm for flexible job shop scheduling with fuzzy
processing time. International Journal of Productions Research, 48(10), 2995–3013.
Lei, D. (2012). Co-evolutionary genetic algorithm for fuzzy flexible job shop scheduling.

Lei, D. (2012). Co-evolutionary genetic algorithm for fuzzy flexible job shop scheduling.
Applied Soft Computing, 12(8), 2237–2245.
Lei, D., Gao, L., & Zheng, Y. (2018). A novel teaching-learning-based optimization

Lei, D., Gao, L., & Zheng, Y. (2018). A novel teaching-learning-based optimization
algorithm for energy-efficient scheduling in hybrid flow shop. IEEE Transactions on
Engineering Management, 65(2), 330–340.
Lei, D., Li, M., & Wang, L. (2019). A two-phase meta-heuristic for multiobjective

Lei, D., Li, M., & Wang, L. (2019). A two-phase meta-heuristic for multiobjective
flexible job shop scheduling problem with total energy consumption threshold. IEEE
Transactions on Cybernetics, 49(3), 1097–1109.
Li, Y., Huang, W., Wu, R., & Guo, K. (2020). An improved artificial bee colony algorithm

Li, J., Liu, Z., Li, C., & Zheng, Z. (2020). Improved artificial immune system algorithm
for type-2 fuzzy flexible job shop scheduling problem. IEEE Transactions on Fuzzy
Systems, 1.
Lin, J. (2015). A hybrid biogeography-based optimization for the fuzzy flexible job-shop

Li, Y., Huang, W., Wu, R., & Guo, K. (2020). An improved artificial bee colony algorithm
for solving multi-objective low-carbon flexible job shop scheduling problem. Applied
Soft Computing, 95, Article 106544.
Li, J., Liu, Z., Li, C., & Zheng, Z. (2020). Improved artificial immune system algorithm

algorithm Palombarini, J. A., & Martínez, E. C. (2019). Closed-loop rescheduling using deep
using reference-point-based nondominated sorting approach, Part I: Solving prob-reinforcement learning. IFAC-PapersOnLine, 52(1), 231–236.
18(4), Pan, Z., Lei, D., & Wang, L. (2021). A bi-population evolutionary algorithm with
feedback for energy-efficient fuzzy flexible job shop scheduling. IEEE Transactions
Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective on Systems, Man, and Cybernetics: Systems, 1–13.
6(2), Pavlov, A. A., Misura, E. B., Melnikov, O. V., & Mukha, I. P. (2019). NP-hard scheduling
problems in planning process automation in discrete systems of certain classes. In
Dorfeshan, Y., Tavakkoli-Moghaddam, R., Mousavi, S. M., & Vahedi-Nouri, B. (2020). A Z. Hu, S. Petoukhov, I. Dychka, & M. He (Eds.), Advances in computer science for
new weighted distance-based approximation methodology for flow shop scheduling engineering and education (pp. 429–436). Cham: Springer International Publishing.
Soft Pericleous, S., Konstantinidis, A., Achilleos, A., & Papadopoulos, G. A. (2017). Generic
hybridization of MOEA/D with learning for permutation flow shop scheduling
Du, Y., Xing, L., Zhang, J., Chen, Y., & He, Y. (2019). MOEA based memetic algorithms problem. In 2017 8th International conference on information, intelligence, systems
Evolutionary & applications (pp. 1–6).
Qin, A. K., Huang, V. L., & Suganthan, P. N. (2009). Differential evolution algorithm
K. with strategy adaptation for global numerical optimization. IEEE Transactions on
job-shop Evolutionary Computation, 13(2), 398–417.
scheduling problem with new job insertion. Expert Systems with Applications, 42(21), Qu, S., Chu, T., Wang, J., Leckie, J., & Jian, W. (2015). A centralized reinforcement
learning approach for proactive scheduling in manufacturing. In 2015 IEEE 20th
X. conference on emerging technologies and factory automation (pp. 1–8).
(2016). An improved artificial bee colony algorithm for flexible job-shop scheduling Qu, S., Wang, J., & Shivani, G. (2016). Learning adaptive dispatching rules for a
problem with fuzzy processing time. Expert Systems with Applications, 65, 52–67. manufacturing process system by using reinforcement learning approach. In 2016
Gao, K. Z., Suganthan, P. N., Pan, Q. K., & Tasgetiren, M. F. (2015). An effective discrete IEEE 21st international conference on emerging technologies and factory automation (pp.
fuzzy 1–8).
processing time. International Journal of Productions Research, 53(19), 5896–5911. Rifai, A. P., Mara, S. T. W., & Sudiarso, A. (2021). Multi-objective distributed reentrant
Gong, W., Liao, Z., Mi, X., Wang, L., & Guo, Y. (2021). Nonlinear equations solving permutation flow shop scheduling with sequence-dependent setup time. Expert
and Systems with Applications, 183, Article 115339.
Sakawa, M., & Kubota, R. (2000). Fuzzy programming for multiobjective job
Han, B. A., & Yang, J. J. (2020). Research on adaptive job shop scheduling problems shop scheduling with fuzzy processing time and fuzzy duedate through genetic
algorithms. European Journal of Operational Research, 120(2), 393–407.
Lang, S., Reggelin, T., Schmidt, J., Müller, M., & Nahhas, A. (2021). NeuroEvolution of Shaheed, I. M., Shukor, S. A., & Abdullah, S. (2018). Population initialisation methods
augmenting topologies for solving a two-stage hybrid flow shop scheduling prob-for fuzzy job-shop scheduling problems: Issues and future trends. International
lem: A comparison of different solution strategies. Expert Systems with Applications, Journal on Advanced Science, Engineering and Information Technology, 8(4–2),
1820–1828.
fuzzy Shahrabi, J., Adibi, M. A., & Mahootchi, M. (2017). A reinforcement learning approach
processing time. International Journal of Productions Research, 48(10), 2995–3013. to parameter estimation in dynamic job shop scheduling. Computers & Industrial
Lei, D. (2012). Co-evolutionary genetic algorithm for fuzzy flexible job shop scheduling. Engineering, 110, 75–82.
Shao, W., Shao, Z., & Pi, D. (2021a). An ant colony optimization behavior-based
optimization MOEA/D for distributed heterogeneous hybrid flow shop scheduling problem under
algorithm for energy-efficient scheduling in hybrid flow shop. IEEE Transactions on nonidentical time-of-use electricity tariffs. IEEE Transactions on Automation Science
and Engineering, 1–16.
multiobjective Shao, W., Shao, Z., & Pi, D. (2021b). Multi-objective evolutionary algorithm based
flexible job shop scheduling problem with total energy consumption threshold. IEEE on multiple neighborhoods local search for multi-objective distributed hybrid flow
shop scheduling problem. Expert Systems with Applications, 183, Article 115453.
Li, Y., Huang, W., Wu, R., & Guo, K. (2020). An improved artificial bee colony algorithm Shiue, Y.-R., Lee, K.-C., & Su, C.-T. (2018). Real-time scheduling for a smart factory
for solving multi-objective low-carbon flexible job shop scheduling problem. Applied using a reinforcement learning approach. Computers & Industrial Engineering, 125,
604–614.
Li, J., Liu, Z., Li, C., & Zheng, Z. (2020). Improved artificial immune system algorithm Sun, L., Lin, L., Gen, M., & Li, H. (2019). A hybrid cooperative coevolution algorithm
for type-2 fuzzy flexible job shop scheduling problem. IEEE Transactions on Fuzzy for fuzzy flexible job shop scheduling. IEEE Transactions on Fuzzy Systems, 27(5),
1008–1022.
Lin, J. (2015). A hybrid biogeography-based optimization for the fuzzy flexible job-shop Van Nostrand, R. C. (2002). Design of experiments using the taguchi approach: 16 steps
to product and process improvement. Technometrics, 44(3), 289.
job-shop Wang, Y.-F. (2020). Adaptive job shop scheduling strategy based on weighted
scheduling problem with fuzzy processing time. Engineering Applications of Artificial Q-learning algorithm. Journal of Intelligent Manufacturing, 31(2), 417–432.
Wang, G., Li, X., Gao, L., & Li, P. (2021). Energy-efficient distributed heterogeneous
Lin, C., Deng, D., Chih, Y., & Chiu, H. (2019). Smart manufacturing scheduling with welding flow shop scheduling problem using a modified MOEA/D. Swarm and
IEEE Transactions on Industrial Evolutionary Computation, 62, Article 100858.
Wang, J.-j., & Wang, L. (2021). A cooperative memetic algorithm with learning-based
for agent for energy-aware distributed hybrid flow-shop scheduling. IEEE Transactions
on Evolutionary Computation, 1. [http://dx.doi.org/10.1109/TEVC.2021.3106168](http://dx.doi.org/10.1109/TEVC.2021.3106168).
Liu, H., Gu, F., & Zhang, Q. (2014). Decomposition of a multiobjective optimization Wang, H.-X., & Yan, H.-S. (2016). An interoperable adaptive scheduling strategy for
problem into a number of simple multiobjective subproblems. IEEE Transactions on knowledgeable manufacturing based on SMGWQ-learning. Journal of Intelligent
Manufacturing, 27(5), 1085–1095.
Lu, C., Gao, L., Yi, J., & Li, X. (2021). Energy-efficient scheduling of distributed flow Wang, L., Zhou, G., Xu, Y., & Liu, M. (2013). A hybrid artificial bee colony algorithm for
shop with heterogeneous factories: A real-world case from automobile industry in the fuzzy flexible job-shop scheduling problem. International Journal of Productions
Research, 51(12), 3593–3608.
Pareto-Waschneck, B., Reichstaller, A., Belzner, L., Altenmüller, T., Bauernhansl, T., Knapp, A.,
energy-efficient & Kyek, A. (2018). Deep reinforcement learning for semiconductor production
scheduling of distributed permutation flow-shop with limited buffers. Robotics and scheduling. In 2018 29th Annual SEMI advanced semiconductor manufacturing
conference (pp. 301–306).
Luo, S. (2020). Dynamic scheduling for flexible job shop with new job insertions by Watkins, C. J. C. H., & Dayan, P. (1992). Technical note: Q-learning. Machine Learning,
8(3), 279–292.
Meng, L., Ren, Y., Zhang, B., Li, J.-Q., Sang, H., & Zhang, C. (2020). MILP modeling and While, L., Hingston, P., Barone, L., & Huband, S. (2006). A faster algorithm for
optimization of energy- efficient distributed flexible job shop scheduling problem. calculating hypervolume. IEEE Transactions on Evolutionary Computation, 10(1),
29–38.
rescheduling Xu, Y., Wang, L., Wang, S.-y., & Liu, M. (2015). An effective teaching-learning-based
reinforcement optimization algorithm for the flexible job-shop scheduling problem with fuzzy
processing time. Neurocomputing, 148, 260–268.

Lin, J. (2019). Backtracking search based hyper-heuristic for the flexible job-shop
scheduling problem with fuzzy processing time. Engineering Applications of Artificial
Intelligence, 77, 186–196.
Lin, C., Deng, D., Chih, Y., & Chiu, H. (2019). Smart manufacturing scheduling with

Lu, C., Huang, Y., Meng, L., Gao, L., Zhang, B., & Zhou, J. (2022). A Paretobased collaborative multi-objective optimization algorithm for energy-efficient
scheduling of distributed permutation flow-shop with limited buffers. Robotics and
Computer-Integrated Manufacturing, 74, Article 102277.
Luo, S. (2020). Dynamic scheduling for flexible job shop with new job insertions by

Lu, C., Gao, L., Yi, J., & Li, X. (2021). Energy-efficient scheduling of distributed flow
shop with heterogeneous factories: A real-world case from automobile industry in
China. IEEE Transactions on Industrial Informatics, 17(10), 6687–6696.
Lu, C., Huang, Y., Meng, L., Gao, L., Zhang, B., & Zhou, J. (2022). A Pareto-

Palombarini, J. A., & Martínez, E. C. (2018). Automatic generation of rescheduling
knowledge in socio-technical manufacturing systems using deep reinforcement
learning. In 2018 IEEE biennial congress of argentina (pp. 1–8).

Lin, C., Deng, D., Chih, Y., & Chiu, H. (2019). Smart manufacturing scheduling with
edge computing using multiclass deep Q network. IEEE Transactions on Industrial
Informatics, 15(7), 4276–4284.
Liu, C., Chang, C., & Tseng, C. (2020). Actor-critic deep reinforcement learning for

* * *

Yuan, S., Li, T., & Wang, B. (2020). A co-evolutionary genetic algorithm for the
two-machine flow shop group scheduling problem with job-related blocking and
transportation times. Expert Systems with Applications, 152, Article 113360.
Zhang, Q., & Li, H. (2007). MOEA/D: a multiobjective evolutionary algorithm based
on decomposition. IEEE Transactions on Evolutionary Computation, 11(6), 712–731.
Zhang, T., Xie, S., & Rose, O. (2017). Real-time job shop scheduling based on simulation
and Markov decision processes. In WSC ’17, Proceedings of the 2017 winter simulation
conference. IEEE Press.

Zhang, J., Zhu, X., & Li, P. (2020). MOEA/D with many-stage dynamical resource
allocation strategy to solution of many-objective OPF problems. International Journal
of Electrical Power & Energy Systems, 120, Article 106050.
Zhao, M., Li, X., Gao, L., Wang, L., & Xiao, M. (2019). An improved Q-learning based
rescheduling method for flexible job-shops with machine failures. In 2019 IEEE
15th international conference on automation science and engineering (pp. 331–337).
Zhu, Z., & Zhou, X. (2021). A multi-objective multi-micro-swarm leadership hierarchybased optimizer for uncertain flexible job shop scheduling problem with job
precedence constraints. Expert Systems with Applications, 182, Article 115214.