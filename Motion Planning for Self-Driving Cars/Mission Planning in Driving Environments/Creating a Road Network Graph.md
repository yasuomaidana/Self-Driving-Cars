# Creating a Road Network Graph

## Graphs

Graph: $G=(V,E)$

A graph is a discrete structure composed of a set of vertices denoted as $V$ and a set of edges denoted as $E$. For the mission planner, each vertex in $V$ will correspond to a given point on the road network, and each edge $E$ will correspond to the road segment that connects any two points in the road network.

### Breadth First Search (BFS)

Algorithm $BFS(G,s,t)$

1. open $\leftarrow$ Queue()
2. closed $\leftarrow$ Set()
3. predecessors $\leftarrow$ Dict()
4. open.enqueue($s$)
5. **while** ! open.isEmpty() **do**:
6. &emsp; $u$ $\leftarrow$ open.dequeue()
7. &emsp;**if** isGoal($u$) **then**
8. &emsp;&emsp;return extractPath($u$, predecessors)
9. &emsp;**for all** $v\in u$.successors()
10. &emsp;&emsp; **if** $v \in$  closed **or** $v \in$ open **then**
11. &emsp;&emsp;&emsp;continue
12. &emsp;&emsp;open.enqueue($v$)
13. &emsp;&emsp;predecessors[$v]$ $\leftarrow u$
14. &emsp;closed.add($u$)
