# Dijkstra's Shortest Path Search

## Algorithm

1. open $\leftarrow$ [MinHeap](#heap-data-structure)()
2. closed $\leftarrow$ Set()
3. predecessors $\leftarrow$ Dict()
4. open.push($s$,0)
5. **while** !open.isEmpty() **do**
6. &emsp; $u, uCost$ $\leftarrow$ open. pop()
7. &emsp; **if** isGoal($u$) **then**
8. &emsp; &emsp; return extractPath($u$, predecessors)
9. &emsp; **for all** $v \in u$.successors()
10. &emsp; &emsp; **if** $v \in$ closed **then**
11. &emsp; &emsp; &emsp; continue
12. &emsp; &emsp; $uvCost$ $\leftarrow$ edgeCost($G, u, v$)
13. &emsp; &emsp; **if** $v \in$ open **then**
14. &emsp; &emsp; &emsp; **if** $uCost$ + $uvCost <$ open[$v$] **then**
15. &emsp; &emsp; &emsp; &emsp; open[$v$] $\leftarrow uCost + uvCost$
16. &emsp; &emsp; &emsp; &emsp; predecessors[$v$] $\leftarrow u$
17. &emsp; &emsp; else
18. &emsp; &emsp; &emsp; open. push($v, uCost + uvCost$)
19. &emsp; &emsp; predecessors[$v$] $\leftarrow u$
20. &emsp; closed. add($u$)

## Heap Data Structure

Heap has the following Properties:

* Complete Binary Tree: A heap tree is a complete binary tree, meaning all levels of the tree are fully filled except possibly the last level, which is filled from left to right. This property ensures that the tree is efficiently represented using an array.
* Heap Property: This property ensures that the minimum (or maximum) element is always at the root of the tree according to the heap type.
* Parent-Child Relationship: The relationship between a parent node at index $i$ and its children is given by the formulas: left child at index $2i+1$ and right child at index $2i+2$ for 0-based indexing of node numbers.
* Efficient Insertion and Removal: Insertion and removal operations in heap trees are efficient. New elements are inserted at the next available position in the bottom-rightmost level, and the heap property is restored by comparing the element with its parent and swapping if necessary. Removal of the root element involves replacing it with the last element and heapifying down.
* Efficient Access to Extremal Elements: The minimum or maximum element is always at the root of the heap, allowing constant-time access.

### Min vs Max heap [see](https://www.geeksforgeeks.org/introduction-to-min-heap-data-structure/)

||Min Heap|Max Heap|
|-|-|-|
|0|In a Min-Heap the key present at the root node must be less than or equal to among the keys present at all of its children.|In a Max-Heap the key present at the root node must be greater than or equal to among the keys present at all of its children.|
|2| In a Min-Heap the minimum key element is present at the root.|In a Max-Heap the maximum key element is present at the root.|
|3|A Min-Heap uses the ascending priority.|A Max-Heap uses the descending priority.|
|4|In the construction of a Min-Heap, the smallest element has priority.|In the construction of a Max-Heap, the largest element has priority.|
|5|In a Min-Heap, the smallest element is the first to be popped from the heap.|In a Max-Heap, the largest element is the first to be popped from the heap.|
