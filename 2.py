def is_feasible(max_cost, num_nodes, distances, queries, conditions):
    # Apply Floyd-Warshall to calculate minimum distances with query adjustments
    floyd = [row[:] for row in distances]
    for u, v, time, cost in queries:
        if cost <= max_cost:
            floyd[u][v] = min(floyd[u][v], time)
            floyd[v][u] = min(floyd[v][u], time)
    
    # Update all pairs shortest path
    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                floyd[i][j] = min(floyd[i][j], floyd[i][k] + floyd[k][j])
    
    # Check all conditions are met
    for a, b, required_time in conditions:
        if floyd[a][b] > required_time:
            return False
    return True

def solve():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    num_nodes = int(data[index])
    index += 1
    num_edges = int(data[index])
    index += 1
    
    infinity_val = float('inf')
    distances = [[infinity_val] * num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes):
        distances[i][i] = 0
    
    for _ in range(num_edges):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        t = int(data[index])
        index += 1
        distances[u][v] = distances[v][u] = t
    
    num_queries = int(data[index])
    index += 1
    queries = []
    for _ in range(num_queries):
        u = int(data[index]) - 1
        index += 1
        v = int(data[index]) - 1
        index += 1
        time = int(data[index])
        index += 1
        cost = int(data[index])
        index += 1
        queries.append((u, v, time, cost))
    
    num_conditions = int(data[index])
    index += 1
    conditions = []
    for _ in range(num_conditions):
        a = int(data[index]) - 1
        index += 1
        b = int(data[index]) - 1
        index += 1
        required_time = int(data[index])
        index += 1
        conditions.append((a, b, required_time))
    
    low, high = 0, 10**9
    answer = -1
    while low <= high:
        mid = (low + high) // 2
        if is_feasible(mid, num_nodes, distances, queries, conditions):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1
    
    if answer == -1:
        print(answer)
        return
    
    valid_ids = [i + 1 for i, query in enumerate(queries) if query[3] <= answer]
    print(len(valid_ids))
    if valid_ids:
        print(' '.join(map(str, valid_ids)))

if __name__ == "__main__":
    solve()
