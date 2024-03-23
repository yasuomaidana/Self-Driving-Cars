# A* Shortest Path Search

A\* search algorithm is a path finding algorithm that finds the single-pair shortest path between the start node(source) and the target node(destination) of a weighted graph. The algorithm not only considers the actual cost from the start node to the current node($g$) but also tries to estimate the cost will take from the current node to the target node using heuristics ($h$). Then it selects the node that has the lowest $f$-value($f=g+h$) to be the next node to move until it hits the target node. Dijkstra’s algorithm is a special case of A\* algorithm where heuristic is 0 for all nodes.

See:

* [wikipedia](https://en.wikipedia.org/wiki/A*_search_algorithm)
* [A* Search Algorithm](https://yuminlee2.medium.com/a-search-algorithm-42c1a13fcf9f) by Claire Lee

## Euclidean Heuristic

* Exploits structure of the problem
* Fast to calculate
* Straight-line distance between two vertices is a useful estimate of true distance along the graph
$$h(v)= \lVert t-v\rVert$$

## Algorithm

1. open $\leftarrow$ MinHeap()
2. closed $\leftarrow$ Set()
3. predecessors $\leftarrow$ Dict()
4. open.push($s$,0)
5. **while** !open.isEmpty() **do**
6. &emsp; $u, uCost$ $\leftarrow$ open.pop()
7. &emsp; **if** isGoal($u$) **then**
8. &emsp; &emsp; return extractPath($u$, predecessors)
9. &emsp; **for all** $v \in u$.successors()
10. &emsp; &emsp; **if** $v \in$ closed **then**
11. &emsp; &emsp; &emsp; continue
12. &emsp; &emsp; $uvCost$ $\leftarrow$ edgeCost($G, u, v$)
13. &emsp; &emsp; **if** $v \in$ open **then**
14. &emsp; &emsp; &emsp; **if** $uCost$ + $uvCost + h(v) <$ open[$v$] **then**
15. &emsp; &emsp; &emsp; &emsp; open[$v$] $\leftarrow uCost + uvCost + h(v)$
16. &emsp; &emsp; &emsp; &emsp; costs[$v$] $\leftarrow uCost + uvCost$
17. &emsp; &emsp; &emsp; &emsp; predecessors[$v$] $\leftarrow u$
18. &emsp; &emsp; else
19. &emsp; &emsp; &emsp; open.push($v, uCost + uvCost+h(v)$)
20. &emsp; &emsp; &emsp; costs[$v$] $\leftarrow uCost + uvCost$
21. &emsp; &emsp; &emsp; predecessors[$v$] $\leftarrow u$
22. &emsp; closed. add($u$)
