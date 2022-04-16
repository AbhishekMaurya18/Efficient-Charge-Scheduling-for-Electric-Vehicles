import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import deque
from collections import defaultdict
import pandas as pd
import seaborn as sns
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# global map for stations and vehicles use

s_map = defaultdict(list)
v_map = defaultdict(list)


# Constants from the spec
EARTH_RADIUS_KM = 6356.752
MAX_CHARGE = 400

MS_IN_SEC = 1000
MS_IN_MINUTE = 60 * MS_IN_SEC
MS_IN_HOUR = 60 * MS_IN_MINUTE


def degree_to_radian(angle):
    return angle * (22/7) / 180.0

# Return the great-circle distance between two points in KM.
"""
the Earth radius R varies from 6356.752 km at the poles to 6378.137 km at the equator. More importantly, the radius of 
curvature of a north-south line on the earth's surface is 1% greater at the poles (≈6399.594 km) than at the 
equator (≈6335.439 km)—so the haversine formula and law of cosines cannot be guaranteed correct to better than 0.5%.
"""


def haversine_dist(lat1, lng1, lat2, lng2):
    lat_rad1 = degree_to_radian(lat1)
    lng_rad1 = degree_to_radian(lng1)
    lat_rad2 = degree_to_radian(lat2)
    lng_rad2 = degree_to_radian(lng2)

    diff_lat = lat_rad2 - lat_rad1
    diff_lng = lng_rad2 - lng_rad1

    u = math.sin(diff_lat / 2.0)
    v = math.sin(diff_lng / 2.0)

    computation = math.asin(math.sqrt(u * u + math.cos(lat_rad1) * math.cos(lat_rad2) * v * v))

    return 2.0 * EARTH_RADIUS_KM * computation


def convert_km_to_ms_travel(distance_km, speed):
    return int(((distance_km / speed) * MS_IN_HOUR) + 0.5)


def ms_to_hours(ms):
    return ms/MS_IN_HOUR


class Station:
    def __init__(self, sid, lat, lon, roc, coc):
        self.sid = sid        # Station id
        self.lat = lat        # Latitude
        self.lon = lon        # longitude
        self.roc = roc        # Rate of per unit charge
        self.coc = coc        # cost of per unit charge
        self.queue = deque()  # waiting_queue


g = nx.Graph()

# Stations co-ordinates
g.add_node('1', pos=(19.2514, 72.8519))
g.add_node('2', pos=(19.1330, 73.2297))
g.add_node('3', pos=(19.2016, 73.1202))
g.add_node('4', pos=(19.3760, 72.8777))
g.add_node('5', pos=(19.4013, 72.5483))
g.add_node('6', pos=(19.1179, 72.7631))
g.add_node('7', pos=(19.1964, 72.5296))
g.add_node('8', pos=(19.0196, 72.5295))
g.add_node('9', pos=(19.0186, 73.1174))
g.add_node('10', pos=(19.3075, 72.6263))

g.add_node('11', pos=(19.1414, 72.8519))
g.add_node('12', pos=(19.1430, 73.0097))
g.add_node('13', pos=(19.3116, 72.9802))
g.add_node('14', pos=(19.2560, 72.6777))
g.add_node('15', pos=(19.2013, 72.9583))
g.add_node('16', pos=(19.2279, 72.7631))
g.add_node('17', pos=(19.3364, 72.5396))
g.add_node('18', pos=(19.1096, 72.6395))
g.add_node('19', pos=(19.1286, 73.1174))
g.add_node('20', pos=(19.3675, 72.6863))

g.add_node('21', pos=(19.3614, 72.8219))
g.add_node('22', pos=(19.0730, 73.0827))
g.add_node('23', pos=(19.0916, 72.8802))
g.add_node('24', pos=(19.2660, 72.9777))
g.add_node('25', pos=(19.3113, 72.7483))
g.add_node('26', pos=(19.0279, 72.7531))
g.add_node('27', pos=(19.1264, 72.5296))
g.add_node('28', pos=(19.2696, 72.5195))
g.add_node('29', pos=(19.0206, 72.9274))
g.add_node('30', pos=(19.1875, 72.6163))

# edges between nodes
g.add_edge('1', '3')
g.add_edge('1', '4')
g.add_edge('1', '6')
g.add_edge('1', '10')
g.add_edge('1', '11')
g.add_edge('1', '13')
g.add_edge('1', '14')
g.add_edge('1', '15')
g.add_edge('1', '16')
g.add_edge('1', '21')
g.add_edge('1', '24')
g.add_edge('1', '25')
g.add_edge('2', '3')
g.add_edge('2', '9')
g.add_edge('2', '15')
g.add_edge('2', '19')
g.add_edge('2', '22')
g.add_edge('3', '12')
g.add_edge('3', '13')
g.add_edge('3', '15')
g.add_edge('3', '19')
g.add_edge('3', '24')
g.add_edge('4', '5')
g.add_edge('4', '13')
g.add_edge('4', '20')
g.add_edge('4', '21')
g.add_edge('4', '24')
g.add_edge('5', '10')
g.add_edge('5', '17')
g.add_edge('5', '20')
g.add_edge('5', '25')
g.add_edge('6', '8')
g.add_edge('6', '11')
g.add_edge('6', '14')
g.add_edge('6', '16')
g.add_edge('6', '18')
g.add_edge('6', '23')
g.add_edge('6', '26')
g.add_edge('6', '30')
g.add_edge('7', '10')
g.add_edge('7', '14')
g.add_edge('7', '16')
g.add_edge('7', '18')
g.add_edge('7', '27')
g.add_edge('7', '28')
g.add_edge('7', '30')
g.add_edge('8', '18')
g.add_edge('8', '26')
g.add_edge('8', '27')
g.add_edge('8', '30')
g.add_edge('9', '19')
g.add_edge('9', '22')
g.add_edge('9', '23')
g.add_edge('9', '29')
g.add_edge('10', '14')
g.add_edge('10', '17')
g.add_edge('10', '20')
g.add_edge('10', '21')
g.add_edge('10', '25')
g.add_edge('10', '28')
g.add_edge('10', '30')
g.add_edge('11', '12')
g.add_edge('11', '15')
g.add_edge('11', '16')
g.add_edge('11', '23')
g.add_edge('11', '26')
g.add_edge('11', '30')
g.add_edge('12', '15')
g.add_edge('12', '16')
g.add_edge('12', '19')
g.add_edge('12', '22')
g.add_edge('12', '23')
g.add_edge('12', '24')
g.add_edge('12', '29')
g.add_edge('13', '21')
g.add_edge('13', '25')
g.add_edge('14', '16')
g.add_edge('14', '25')
g.add_edge('14', '28')
g.add_edge('14', '30')
g.add_edge('15', '16')
g.add_edge('15', '24')
g.add_edge('16', '18')
g.add_edge('16', '25')
g.add_edge('16', '30')
g.add_edge('17', '20')
g.add_edge('17', '28')
g.add_edge('17', '30')
g.add_edge('18', '26')
g.add_edge('18', '27')
g.add_edge('18', '30')
g.add_edge('19', '15')
g.add_edge('19', '22')
g.add_edge('19', '23')
g.add_edge('20', '21')
g.add_edge('20', '25')
g.add_edge('21', '24')
g.add_edge('21', '25')
g.add_edge('22', '11')
g.add_edge('22', '23')
g.add_edge('22', '29')
g.add_edge('23', '15')
g.add_edge('23', '26')
g.add_edge('23', '29')
g.add_edge('24', '13')
g.add_edge('24', '25')
g.add_edge('26', '27')
g.add_edge('26', '29')
g.add_edge('27', '30')
g.add_edge('28', '30')
g.add_edge('29', '18')

nx.draw(g, nx.get_node_attributes(g, 'pos'), with_labels=True, node_size=250)
plt.savefig("Network_Graph.png")


# Reserve station for refueling
def reserve_according_to_cost():
    # explore the vehicle map
    for key in v_map:
        v_id = key
        reach = v_map[key]

        # A list to contain all possible station having minimum equal cost of per unit charge
        solver = deque([])
        solver.clear()

        for i in range(0, len(reach)):
            if len(solver) == 0:
                solver.append(reach[i])
            else:
                if solver and int((solver[-1][1]-reach[i][1])*100) >= 0:
                    while solver and int((solver[-1][1]-reach[i][1])*100) > 0:
                        solver.pop()
                    solver.append(reach[i])

        # # print the affordable station to go
        # print("\nVehicle " + str(v_id) + " can recharge its battery at Station :\n")
        #
        # size = len(solver)
        # for _ in range(size):
        #     temp = solver[0]  # minimum cost at front
        #     solver.popleft()
        #     print("\t\t" + str(temp[0]) + " in " + str(temp[2]), end=" ")
        #     print("hours, where cost of per unit charge is " + str(temp[1]) + "\n")
        #     # solver.append(temp)

    # print()

    return


def reachable_station(request):
    v_map.clear()
    s_map.clear()

    for i in range(len(request)):
        vid = request[i][0]
        v_lat = request[i][1]
        v_lon = request[i][2]
        v_avg_speed = request[i][3]
        v_soc = request[i][4]

        # print("\nVehicle " + str(vid) + ":\n")

        for j in range(30):
            s_id = station[j]
            dist = haversine_dist(v_lat, v_lon, s_id.lat, s_id.lon)

            if dist > v_soc:
                continue
            else:
                # time = dist / v_avg_speed

                # round of time to 2 decimal places
                # time = (time * 1000 + .5) / 1000.0

                # print(" is " + str(time) + " hours away from Station " + str(s_id.sid) + "\n")

                v_map[vid].append([s_id.sid, s_id.coc, dist])
                s_map[s_id.sid].append([vid, dist])


# index is passed to sort the reachable stations according to index
def wait_count(index, message, requests):
    # explore the vehicle map and waiting time
    for key in v_map:
        v_id = key
        reach = v_map[key]

        # A list to contain all possible station having minimum equal coc and wait queue time
        solver = [[]]
        solver.clear()

        for i in range(len(reach)):
            s = reach[i][0]
            distance = reach[i][2]

            s_id = station[s - 1]

            wait_time = 0
            if len(s_id.queue) > 0:
                wait_time = s_id.queue[-1][1]

            solver.append([s_id, distance, s_id.coc, wait_time])

        # sort the vector solver by wait time
        solver.sort(key=lambda x: x[index])

        # print the affordable station to go

        # print("\nVehicle " + str(v_id) + " can recharge its battery at Station :\n")

        size = len(solver)
        # for i in range(size):
        #     temp = solver[i]
            # print("\t\t" + str(temp[0].sid) + " in " + str(temp[1]), end=" ")
            # print("hours, where cost of per unit charge is " + str(temp[2]) + " and wait time is: " + str(temp[3]))

        # insert reserved vehicle vid in waiting queue at station sid.id
        s_id = solver[0][0]
        wt = request[v_id - 1][4] / s_id.roc  # time to charge vehicle vid at station sid

        if len(s_id.queue) > 0:
            wt += s_id.queue[-1][1]

        s_id.queue.append([v_id, wt])

    # array to store waiting queue length at each station
    wait = []

    for i in range(30):
        s_id = station[i]

        if len(s_id.queue):
            wait.append(len(s_id.queue) - 1)  # -1, since first vehicle is not waiting
        else:
            wait.append(0)

    print(wait)

    # dataset is wait list
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # there is only 10 stations

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(s, wait, color='maroon', width=0.4)

    plt.xlabel("Station ID")
    plt.ylabel("Waiting Queue length")
    plt.title("WQL at different Stations using " + message + " with " + str(requests) + " requests")
    plt.show()

    # return the waiting queues length
    return wait


def fuzzy_logic_wq_dist_cost(wql, d, price):
    wait = ctrl.Antecedent(np.arange(0, 60, 1), 'waiting queue length')
    dist = ctrl.Antecedent(np.arange(0, 100, 1), 'distance')
    cost = ctrl.Antecedent(np.arange(8, 16, 0.2), 'cost')
    result = ctrl.Consequent(np.arange(0, 101, 1), 'result')

    wait['poor'] = fuzz.trimf(wait.universe, [0, 0, 25])
    wait['average'] = fuzz.trimf(wait.universe, [8, 25, 45])
    wait['good'] = fuzz.trimf(wait.universe, [30, 60, 60])

    dist['poor'] = fuzz.trimf(dist.universe, [0, 25, 50])
    dist['average'] = fuzz.trimf(dist.universe, [25, 50, 75])
    dist['good'] = fuzz.trimf(dist.universe, [50, 75, 100])

    cost['poor'] = fuzz.trimf(cost.universe, [8, 10, 12])
    cost['average'] = fuzz.trimf(cost.universe, [10, 12, 14])
    cost['good'] = fuzz.trimf(cost.universe, [12, 14, 16])

    result['low'] = fuzz.trimf(result.universe, [0, 25, 50])
    result['medium'] = fuzz.trimf(result.universe, [25, 50, 75])
    result['high'] = fuzz.trimf(result.universe, [50, 100, 100])

    # wait['average'].view()
    # cost['average'].view()
    # dist['average'].view()
    #
    # result.view()

    rule1 = ctrl.Rule(wait['poor'] & (cost['poor'] | dist['poor']), result['high'])
    rule2 = ctrl.Rule(wait['poor'] & (cost['poor'] | dist['average']), result['high'])
    rule3 = ctrl.Rule(wait['poor'] & (cost['poor'] | dist['good']), result['high'])

    rule4 = ctrl.Rule(wait['poor'] & (cost['average'] | dist['poor']), result['high'])
    rule5 = ctrl.Rule(wait['poor'] & (cost['average'] | dist['average']), result['high'])
    rule6 = ctrl.Rule(wait['poor'] & cost['average'] & dist['good'], result['high'])

    rule7 = ctrl.Rule(wait['poor'] & (cost['good'] | dist['poor']), result['high'])
    rule8 = ctrl.Rule(wait['poor'] & cost['good'] & dist['average'], result['high'])
    rule9 = ctrl.Rule(wait['poor'] & cost['good'] & dist['good'], result['high'])

    rule10 = ctrl.Rule(wait['average'] & cost['poor'] & dist['poor'], result['high'])
    rule11 = ctrl.Rule(wait['average'] & cost['poor'] & dist['average'], result['medium'])
    rule12 = ctrl.Rule(wait['average'] & cost['poor'] & dist['good'], result['medium'])

    rule13 = ctrl.Rule(wait['average'] & cost['average'] & dist['poor'], result['medium'])
    rule14 = ctrl.Rule(wait['average'] & cost['average'] & dist['average'], result['medium'])
    rule15 = ctrl.Rule(wait['average'] & cost['average'] & dist['good'], result['medium'])

    rule16 = ctrl.Rule(wait['average'] & cost['good'] & dist['poor'], result['medium'])
    rule17 = ctrl.Rule(wait['average'] & cost['good'] & dist['average'], result['medium'])
    rule18 = ctrl.Rule(wait['average'] & cost['good'] & dist['good'], result['low'])

    rule19 = ctrl.Rule(wait['good'] & cost['poor'] & dist['poor'], result['low'])
    rule20 = ctrl.Rule(wait['good'] & cost['poor'] & dist['average'], result['low'])
    rule21 = ctrl.Rule(wait['good'] & cost['poor'] & dist['good'], result['low'])

    rule22 = ctrl.Rule(wait['good'] & cost['average'] & dist['poor'], result['low'])
    rule23 = ctrl.Rule(wait['good'] & cost['average'] & dist['average'], result['low'])
    rule24 = ctrl.Rule(wait['good'] & cost['average'] & dist['good'], result['low'])

    rule25 = ctrl.Rule(wait['good'] & cost['good'] & dist['poor'], result['low'])
    rule26 = ctrl.Rule(wait['good'] & cost['good'] & dist['average'], result['low'])
    rule27 = ctrl.Rule(wait['good'] & cost['good'] & dist['good'], result['low'])

    result_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11,
                                      rule12, rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21,
                                      rule22, rule23, rule24, rule25, rule26, rule27])

    res = ctrl.ControlSystemSimulation(result_ctrl)

    res.input['waiting queue length'] = wql
    res.input['distance'] = d
    res.input['cost'] = price
    res.compute()

    # print(res.output['result'])
    output = "{0:.2f}".format(res.output['result'])

    # result.view(sim=res)
    return output


# sort the reachable stations according to fuzzy logic
def fuzzy_wait_count(index, x):
    # explore the vehicle map and waiting time
    for key in v_map:
        v_id = key
        reach = v_map[key]

        # A list to contain all possible station having minimum equal coc and wait queue time
        solver = []
        solver.clear()

        for i in range(len(reach)):
            s = reach[i][0]
            distance = reach[i][2]

            s_id = station[s - 1]

            wait_time = 0
            if len(s_id.queue) > 0:
                wait_time = s_id.queue[-1][1]

            solver.append([s_id, distance, s_id.coc, wait_time])
        #
        # # sort the vector solver by wait time
        # solver.sort(key=lambda x: (x[2], x[3], x[1]))

        support_vec = []
        support_vec.clear()

        for element in solver:
            ans = 0
            if x == 4:
                ans = fuzzy_logic_wq_dist_cost(len(element[0].queue), element[1], element[2])  # fuzzy function call
            support_vec.append([element[0], ans])

        support_vec.sort(key=lambda y: y[1])

        # print(solver)

        size = len(solver)

        # insert reserved vehicle vid in waiting queue at station sid.id
        s_id = support_vec[-1][0]
        wt = 0    # request[v_id - 1][4] / s_id.roc, time to charge vehicle vid at station sid

        if len(s_id.queue) > 0:
            wt += s_id.queue[-1][1]

        s_id.queue.append([v_id, wt])

    # array to store waiting queue length at each station
    wait = []

    for i in range(30):
        s_id = station[i]

        if len(s_id.queue):
            wait.append(len(s_id.queue) - 1)  # -1, since first vehicle is not waiting
        else:
            wait.append(0)

    print(wait)

    # dataset is wait list
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    # there is only 30 stations

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(s, wait, color='maroon', width=0.4)

    plt.xlabel("Station ID")
    plt.ylabel("Waiting Queue length")
    plt.title("WQL at different Stations using Fuzzy Logic with " + str(index) + " requests")
    plt.show()

    return wait


def clear_wait_queue(st):

    for s_name in st:
        s_name.queue.clear()


def plot_line_chart(wq1, wq2, wq3, wq4):

    # creating dataframe of station id and waiting queues
    df = pd.DataFrame({'Station id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                                      24, 25, 26, 27, 28, 29, 30],
                       'Wait': wq1,
                       'Dist': wq2,
                       'Cost': wq3,
                       'Fuzzy_L': wq4})

    # convert to long (tidy) form
    dfm = df.melt('Station id', var_name='cols', value_name='Waiting Queue Length')  # wql=wait queue length

    sns.pointplot(x="Station id", y="Waiting Queue Length", hue="cols", data=dfm)


if __name__ == "__main__":

    # creating station specification
    s1 = Station(1, 19.2514, 72.8519, 131.0, 9.5)
    s2 = Station(2, 19.1330, 73.2297, 159.0, 11.5)
    s3 = Station(3, 19.2016, 73.1202, 133.0, 12.1)
    s4 = Station(4, 19.3760, 72.8777, 143.0, 10.4)
    s5 = Station(5, 19.4013, 72.5483, 105.0, 13.5)
    s6 = Station(6, 19.1179, 72.7631, 175.0, 14.2)
    s7 = Station(7, 19.1964, 72.5296, 153.0, 15.2)
    s8 = Station(8, 19.0196, 72.5295, 106.0, 10.7)
    s9 = Station(9, 19.0186, 73.1174, 114.0, 11.1)
    s10 = Station(10, 19.3075, 72.6263, 116.0, 12)

    s11 = Station(11, 19.1414, 72.8519, 131.0, 11.5)
    s12 = Station(12, 19.1430, 73.0097, 159.0, 10.5)
    s13 = Station(13, 19.3116, 72.9802, 133.0, 12.1)
    s14 = Station(14, 19.2560, 72.6777, 143.0, 10.4)
    s15 = Station(15, 19.2013, 72.9583, 105.0, 14.5)
    s16 = Station(16, 19.2279, 72.7631, 175.0, 15.2)
    s17 = Station(17, 19.3364, 72.5396, 153.0, 15.2)
    s18 = Station(18, 19.1096, 72.6395, 106.0, 8.7)
    s19 = Station(19, 19.1286, 73.1174, 114.0, 11.1)
    s20 = Station(20, 19.3675, 72.6863, 116.0, 12)

    s21 = Station(21, 19.3614, 72.8219, 131.0, 12.5)
    s22 = Station(22, 19.0730, 73.0827, 159.0, 13.5)
    s23 = Station(23, 19.0916, 72.8802, 133.0, 11.1)
    s24 = Station(24, 19.2660, 72.9777, 143.0, 10.4)
    s25 = Station(25, 19.3113, 72.7483, 105.0, 9.5)
    s26 = Station(26, 19.0279, 72.7531, 175.0, 14.2)
    s27 = Station(27, 19.1264, 72.5296, 153.0, 12.2)
    s28 = Station(28, 19.2696, 72.5195, 106.0, 10.7)
    s29 = Station(29, 19.0206, 72.9274, 114.0, 11.1)
    s30 = Station(30, 19.1875, 72.6163, 116.0, 14)

    station = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18, s19, s20,
               s21, s22, s23, s24, s25, s26, s27, s28, s29, s30]

    # vehicles recharge request [vehicle id, latitude, longitude, avg speed, soc]

    request = [[1, 18.59, 72.98, 71, 220], [2, 18.88, 71.8, 72, 181], [3, 18.8, 71.89, 65, 174],
               [4, 20.5, 73.8, 67, 162], [5, 18.7, 71.9, 50, 87], [6, 18.7, 72.9, 60, 95],
               [7, 18.7, 73.02, 70, 110], [8, 19.3, 71.7, 47, 200], [9, 19.58, 72.9, 70, 210],
               [10, 18.88, 72.18, 72, 181], [11, 18.78, 71.9, 65, 174], [12, 20.45, 73, 67, 162],
               [13, 18.67, 71.9, 50, 87], [14, 18.667, 72.9, 60, 95], [15, 18.7, 73.03, 70, 100],
               [16, 19.28, 72.7, 47, 200], [17, 18.5, 72.98, 70, 220], [18, 18.78, 71.8, 72, 181],
               [19, 18.88, 71.89, 65, 174], [20, 20.3, 72.8, 67, 162], [21, 18.67, 71.9, 50, 87],
               [22, 18.7, 73.22, 60, 95], [23, 18.71, 73.02, 70, 110], [24, 19.31, 72.7, 47, 200],
               [25, 18.58, 73.21, 70, 210], [26, 18.88, 73.28, 72, 181], [27, 19.8, 71.9, 65, 174],
               [28, 20.46, 73, 67, 162], [29, 18.16, 71.9, 50, 187], [30, 18.717, 72.89, 60, 95],
               [31, 18.71, 72.03, 70, 100], [32, 19.3, 72.7, 47, 200], [33, 19.32, 71.7, 47, 205],
               [34, 18.5, 72.98, 70, 220], [35, 18.87, 71.81, 72, 181], [36, 18.8, 71.891, 65, 170],
               [37, 20.51, 72.8, 67, 160], [38, 18.72, 71.9, 50, 89], [39, 18.72, 71.9, 60, 90],
               [40, 18.72, 73.12, 70, 110], [41, 19.32, 71.7, 47, 210], [42, 19.6, 72.9, 70, 210],
               [43, 18.8, 71.818, 72, 180], [44, 19.7, 71.9, 65, 175], [45, 20.465, 73, 67, 160],
               [46, 18.613, 71.9, 50, 80], [47, 18.617, 72.92, 60, 95], [48, 18.71, 72.03, 70, 150],
               [49, 19.38, 72.7, 47, 200], [50, 19.52, 73.38, 70, 200], [51, 18.78, 71.83, 72, 186],
               [52, 18.88, 71.85, 65, 170], [53, 20.33, 72.8, 67, 168], [54, 18.73, 71.9, 50, 97],
               [55, 18.71, 72.92, 60, 105], [56, 18.71, 73.12, 70, 120], [57, 19.35, 72.7, 47, 200],
               [58, 18.58, 72.92, 70, 200], [59, 18.88, 71.618, 72, 180], [60, 18.82, 71.91, 65, 174],
               [61, 20.426, 73, 67, 162], [62, 18.716, 71.912, 50, 89], [63, 18.7127, 72.89, 60, 95],
               [64, 18.712, 72.03, 70, 100], [65, 19.33, 72.7, 47, 200], [66, 19.321, 72.7, 47, 205],
               [67, 19.591, 71.98, 70, 220], [68, 17.539, 71.98, 70, 220], [69, 19.88, 71.778, 72, 181],
               [70, 18.7568, 71.821, 72, 181], [71, 18.88, 71.82, 72, 181], [72, 19.8, 71.9, 65, 174],
               [73, 20.54, 72.8, 67, 162], [74, 18.7, 71.912, 50, 87], [75, 18.712, 73.09, 60, 95],
               [76, 18.714, 72.02, 70, 110], [77, 19.3, 72.712, 47, 200], [78, 19.585, 73.19, 70, 210],
               [79, 18.88, 71.7178, 72, 181], [80, 19.78, 71.912, 65, 174], [81, 20.45, 73.13, 67, 150],
               [82, 18.67127, 71.91, 50, 87], [83, 18.717, 73.2945, 60, 95], [84, 18.765, 72.23, 70, 100],
               [85, 19.278, 72.7, 47, 200], [86, 19.55, 73.18, 70, 220], [87, 18.78, 71.823, 72, 181],
               [88, 18.865, 71.89, 65, 174], [89, 20.36, 73.18, 67, 162], [90, 18.67, 71.95, 50, 87],
               [91, 18.723, 73.192, 60, 95], [92, 18.71, 72.022, 70, 110], [93, 19.315, 71.7, 47, 200],
               [94, 19.58, 73.1913, 70, 210], [95, 19.88, 71.8238, 72, 181], [96, 18.8, 71.936, 65, 174],
               [97, 19.46, 73.15, 67, 162], [98, 18.7167, 71.9, 50, 87], [99, 18.617, 73.1879, 60, 95],
               [100, 18.71, 72.043, 70, 100], [101, 19.32, 72.71, 47, 210], [102, 19.32, 72.723, 47, 205],
               [103, 19.595, 71.98, 70, 220], [104, 18.88, 71.823, 72, 181], [105, 18.878, 71.89, 65, 174],
               [106, 19.5, 72.856, 67, 162], [107, 18.7, 71.815, 50, 87], [108, 18.7, 72.876, 60, 95],
               [109, 18.687, 73.102, 70, 110], [110, 19.287, 72.7, 47, 200], [111, 19.58, 71.911, 70, 210],
               [112, 18.868, 71.8018, 72, 181], [113, 18.78, 71.901, 65, 174], [114, 19.455, 73.1, 67, 162],
               [115, 18.66167, 71.9, 50, 87], [116, 18.7717, 73.1789, 60, 95], [117, 18.7, 73.0113, 70, 170],
               [118, 19.28, 72.687, 47, 200], [119, 19.511, 73.198, 70, 220], [120, 19.78, 71.8012, 72, 181],
               [121, 18.856, 71.89, 65, 174], [122, 20.3, 73.18012, 67, 162], [123, 18.675, 71.9, 50, 87],
               [124, 18.7012, 72.92, 60, 95], [125, 18.71, 73.023, 70, 110], [126, 19.31, 72.712, 47, 200],
               [127, 19.58, 72.871, 70, 210], [128, 18.88, 71.8278, 72, 181], [129, 18.81, 71.911, 65, 174],
               [130, 20.461, 73.12, 67, 162], [131, 18.716, 71.9013, 50, 87], [132, 18.617, 72.8669, 60, 95],
               [133, 18.71, 72.0123, 70, 140], [134, 19.3, 72.741, 47, 200], [135, 19.32, 72.723, 47, 205],
               [136, 20.444, 73.8, 67, 162], [137, 18.7, 71.769, 50, 157], [138, 18.7, 73.11901, 60, 95],
               [139, 18.7, 73.0022, 70, 120], [140, 19.3, 72.667, 47, 200], [141, 19.58, 73.0849, 70, 210],
               [142, 18.88, 71.71238, 72, 181], [143, 19.765, 71.9, 65, 174], [144, 20.45, 73.001, 67, 162],
               [145, 18.6717, 71.569, 50, 87], [146, 18.75170, 73.09, 60, 95], [147, 18.7, 72.032, 70, 100],
               [148, 19.2128, 72.7, 47, 200], [149, 19.502, 73.0918, 70, 220], [150, 18.78, 71.8018, 72, 181]
               ]

    # for i in range(6):
    #     s_id = station[i]
    #     s_id.queue.append([(i+21), 2])     # append {vid, charging time} in waiting queue

    # store the waiting queue length
    wl_50_wait, wl_50_dist, wl_50_cost = [], [], []
    wl_100_wait, wl_100_dist, wl_100_cost = [], [], []
    wl_150_wait, wl_150_dist, wl_150_cost = [], [], []

    # clear the waiting queue
    clear_wait_queue(station)

    # find vehicle reach and wait list and draw graph
    for x in range(3):
        if x == 0:
            # find reachable station for vehicle
            reachable_station(request[:50])

            # waiting queue length count wl_50_wait=wait queue length when request is 50 and wait is considered
            wl_50_wait = wait_count(3, "wait", 50)   # 3 because in reachable station at index 3 wait time is stored
            clear_wait_queue(station)

            # waiting queue length count
            wl_50_dist = wait_count(1, "distance", 50)  # 1 because in reachable station at index 1 time = dist/avg_speed is stored
            clear_wait_queue(station)

            # waiting queue length count
            wl_50_cost = wait_count(2, "cost", 50)  # 2 because in reachable station at index 2 cost of per unit charge is stored
            clear_wait_queue(station)

            wl_50_FL = fuzzy_wait_count(50, 4)

            plot_line_chart(wl_50_wait, wl_50_dist, wl_50_cost, wl_50_FL)

            wl_50_wait = wait_count(3, "wait", 50)

            # clear the waiting queue
            clear_wait_queue(station)
            print()
        # elif x == 1:
        #     reachable_station(request[:100])
        #
        #     # waiting queue length count
        #     wl_100_wait = wait_count(3, "wait", 100)
        #     clear_wait_queue(station)
        #
        #     wl_100_dist = wait_count(1, "distance", 100)  # 1 because in reachable station at index 1 time = dist/avg_speed is stored
        #     clear_wait_queue(station)
        #
        #     wl_100_cost = wait_count(2, "cost", 100)  # 2 because in reachable station at index 2 cost of per unit charge is stored
        #     clear_wait_queue(station)
        #
        #     wl_100_FL = fuzzy_wait_count(100, 4)
        #
        #     plot_line_chart(wl_100_wait, wl_100_dist, wl_100_cost, wl_100_FL)
        #
        #     # clear the waiting queue
        #     clear_wait_queue(station)
        #     print()
        # else:
        #     reachable_station(request[:])
        #
        #     # waiting queue length count
        #     wl_150_wait = wait_count(3, "wait", 150)  # 3 because in reachable station at index 3 wait time is stored
        #     clear_wait_queue(station)
        #
        #     wl_150_dist = wait_count(1, "distance", 150)  # 1 because in reachable station at index 1 time = dist/avg_speed is stored
        #     clear_wait_queue(station)
        #
        #     wl_150_cost = wait_count(2, "cost", 150)  # # 2 because in reachable station at index 2 cost of per unit charge is stored
        #     clear_wait_queue(station)
        #
        #     wl_150_FL = fuzzy_wait_count(150, 4)
        #
        #     plot_line_chart(wl_150_wait, wl_150_dist, wl_150_cost, wl_150_FL)
        #     wl_150_dist = wait_count(1, "distance", 150)
        #
        #     # clear the waiting queue
        #     clear_wait_queue(station)











    #     # # reserve station according to minimum cost
    #     # reserve()
    #
    # # print(len(s3.queue))
    #
    # # clear the waiting queue
    # clear_wait_queue(station)

    # print(len(wl_50_wait), len(wl_50_dist), len(wl_50_cost))
    # print(len(wl_100_wait), len(wl_100_dist), len(wl_100_cost))
    # print(len(wl_150_wait), len(wl_150_dist), len(wl_150_cost))










