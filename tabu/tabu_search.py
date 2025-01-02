
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

# 节点类
class Node:
    def __init__(self):
        self.node_id = 0
        self.x = 0.0
        self.y = 0.0


# 解
class Solution:
    def __init__(self):
        self.cost = 0.0
        self.route = []

    def __copy__(self):
        solution = Solution()
        solution.cost = self.cost
        solution.route = self.route
        return solution


def route_cost(route, distance, city_number):
    cost = 0
    for index in range(city_number - 1):
        cost += distance[route[index]][route[index + 1]]
    cost += distance[route[city_number - 1]][route[0]]
    return cost


def swap(route, route_i, route_j):
    route_ten = len(route)
    if route_i < 0 or route_i >= int(route_ten) or route_j < 0 or route_j >= int(route_ten):
        return route
    new_route = copy.deepcopy(route)
    new_route[route_i] = route[route_j]
    new_route[route_j] = route[route_i]
    return new_route


def reverse(route, route_i, route_j):
    new_route = copy.deepcopy(route)
    while route_i < route_j:
        route_temp = new_route[route_i]
        new_route[route_i] = new_route[route_j]
        new_route[route_j] = route_temp
        route_i += 1
        route_j -= 1
    return new_route


def  get_candidate(route, distance, city_number, candidate_number):
    candidate = [] # 候选解的集合
    candidate_cost = [] # 候选解的成本的集合
    swap_position = []
    i = 0
    while i < candidate_number:
        current_position = random.sample(range(0, city_number), 2)
        if current_position not in swap_position:
            # print("current_position" + str(current_position))
            swap_position.append(current_position)
            candidate.append(swap(route, current_position[0], current_position[1]))
            candidate_cost.append(route_cost(candidate[i], distance, city_number))
            i += 1
    return candidate, candidate_cost, swap_position

if __name__ == '__main__':
    print("tabu_search start!")

    # 生成地图的节点集合
    N = [] # 节点集合
    size = 20 #
    maxX = maxY = 100
    for i in range(0, size):
        node = Node()
        node.id = i
        node.x = random.randint(1, maxX)
        node.y = random.randint(1,maxY)
        N.append(node)

    # 计算距离矩阵
    distance = [[0 for j in range(size)] for i in range(size)]
    for i in range(size):
        for j in range(size):
            distance[i][j] = ((N[i].x - N[j].x) ** 2 + (N[i].y - N[j].y) ** 2) ** 0.05
            distance[j][i] = distance[i][j]

    # 配置算法配置
    Iter = 5000 # 迭代次数
    city_number = size # 城市数量

    tabu_list = []                  # 禁忌表
    tabu_del_list = []              # 从禁忌表中要删除的部分
    tabu_length = city_number * 2   # 禁忌表的长度

    history_beat_route = []         # 每次迭代中好的解
    history_best_cost = []          # 每次迭代中最好的解的成本

    best_solution = Solution()      # 所有迭代中最好的解
    best_solution.cost = 9999       # 所有迭代中最好的解的成本，赋予一个最大值
    # 当前初始解的初始化
    current_solution = Solution()
    current_solution.route = list(range(size))
    current_solution.cost = route_cost(current_solution.route, distance, city_number)

    # 算法部分
    for k in range(Iter): # 迭代Iter次数
        print("第 " + str(k) + " 次迭代")
        candidate, candidate_distance, swap_position = get_candidate(current_solution.route, distance, city_number, candidate_number=50)
        min_index = np.argmin(candidate_distance)
        if swap_position[min_index] not in tabu_list: # 这个解不在禁忌表中
            # 情形1：候选解的成本比当前最优解更好
            if candidate_distance[min_index] < best_solution.cost:
                best_solution.route = candidate[min_index].copy()
                best_solution.cost = candidate_distance[min_index]

            current_solution.route = candidate[min_index].copy()
            current_solution.cost = candidate_distance[min_index]
            # 插入禁忌表
            tabu_list.append(swap_position[min_index])
            # 插入过程记录
            history_beat_route.append(best_solution.route)
            history_best_cost.append(best_solution.cost)
        else: # 在紧急表中
            # 情形3：候选解的成本比当前最优解更好
            if candidate_distance[min_index] < best_solution.cost:
                tabu_del_list.append(tabu_list.index(swap_position[min_index]))
                del tabu_list[tabu_list.index(swap_position[min_index])]
                best_solution.route = candidate[min_index].copy()
                best_solution.cost = candidate_distance[min_index]
            else:
                candidate_distance[min_index] = 9999

            current_solution.route = candidate[min_index].copy()
            current_solution.cost = candidate_distance[min_index]
            # 插入过程记录
            history_beat_route.append(best_solution.route)
            history_best_cost.append(best_solution.cost)
        if len(tabu_list) > tabu_length:
            del tabu_list[0]

    # print(history_beat_route)
    # print(history_best_cost)

    plt.figure(1)
    x_axis_data = list(range(Iter)) # x
    y_axis_data = history_best_cost  # y
    plt.plot(x_axis_data, y_axis_data, 'b*--', alpha=0.5, linewidth=1, label='acc')  # 'bo-'表示蓝色实线，数据点实心原点标注
    ## plot中参数的含义分别是横轴值，纵轴值，线的形状（'s'方块,'o'实心圆点，'*'五角星   ...，颜色，透明度, 线的宽度和标签 ，
    plt.legend("解的成本")  # 显示上面的label
    plt.xlabel('迭代次数')  # x_label
    plt.ylabel('成本')  # y_label

    plt.figure(2)
    x_axis_data_2, y_axis_data_2 = [], []
    for i in best_solution.route:
        x_axis_data_2.append(N[i].x)
        y_axis_data_2.append(N[i].y)
    x_axis_data_2.append(N[best_solution.route[0]].x)
    y_axis_data_2.append(N[best_solution.route[0]].y)
    plt.plot(x_axis_data_2, y_axis_data_2, 'b*--')
    plt.xlim(-1, maxX)  # 仅设置x轴坐标范围
    plt.ylim(-1, maxY)  # 仅设置y轴坐标范围
    plt.show()

    # plt.ylim(-1,1)#仅设置y轴坐标范围



    print("tabu_search end!")




