import numpy as np
import random

def Chinese_Restaurant_Process(num_customer, alpha):
    if num_customer == 0:
        return []
    table_assignments = [1]
    next_open_table = 2
    for i in range(1, num_customer - 1):
        prob = float(alpha) / (alpha + i)
        rand = random.uniform(0, 1)
        if rand < prob:
            table_assignments.append(next_open_table)
            next_open_table = next_open_table + 1
        else:
            randId = int(random.uniform(0, len(table_assignments)))
            which_table = table_assignments[randId]
            table_assignments.append(which_table)
    return table_assignments

if __name__ == '__main__':
    print Chinese_Restaurant_Process(10, 3)
