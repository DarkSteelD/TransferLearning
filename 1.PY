from collections import deque

num, window = map(int, input().split())
costs = [int(x) for x in input().split()]
prize_counts = [0] * num
total_min_cost = 0
min_indices = deque()

for current_index in range(num):
    if min_indices and min_indices[0] < current_index - window + 1:
        min_indices.popleft()
    while min_indices and costs[min_indices[-1]] > costs[current_index]:
        min_indices.pop()
    min_indices.append(current_index)
    if current_index >= 0:  # This checks ensures prizes are calculated from day one
        min_cost_index = min_indices[0]
        min_cost = costs[min_cost_index]
        prize_counts[min_cost_index] += 1
        total_min_cost += min_cost

print(total_min_cost)
print(' '.join(map(str, prize_counts)))
