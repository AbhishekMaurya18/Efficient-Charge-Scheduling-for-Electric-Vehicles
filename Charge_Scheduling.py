import networkx as nx
import matplotlib.pyplot as plt
import math
from collections import deque
from collections import defaultdict


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

g.add_node('1', pos=(19.2514, 72.8519))
g.add_node('2', pos=(19.1330, 73.2297))
g.add_node('3', pos=(19.2016, 73.1202))
g.add_node('4', pos=(19.3760, 72.8777))
g.add_node('5', pos=(19.4013, 72.5483))
g.add_node('6', pos=(19.1179, 72.7631))
g.add_node('7', pos=(19.2364, 72.5296))
g.add_node('8', pos=(19.0596, 72.5295))
g.add_node('9', pos=(19.0186, 73.1174))
g.add_node('10', pos=(19.3075, 72.6263))

g.add_edge('1', '3')
g.add_edge('1', '4')
g.add_edge('1', '5')
g.add_edge('1', '6')
g.add_edge('1', '7')
g.add_edge('2', '3')
g.add_edge('2', '6')
g.add_edge('2', '9')
g.add_edge('3', '4')
g.add_edge('3', '6')
g.add_edge('3', '9')
g.add_edge('4', '5')
g.add_edge('4', '10')
g.add_edge('5', '7')
g.add_edge('5', '10')
g.add_edge('6', '9')
g.add_edge('6', '8')
g.add_edge('6', '10')
g.add_edge('7', '6')
g.add_edge('7', '8')
g.add_edge('8', '9')
g.add_edge('10', '1')
g.add_edge('10', '7')


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



    print()

    return


def reachable_station( request ):
    v_map.clear()
    s_map.clear()

    for i in range(len(request)):
        vid = request[i][0]
        v_lat = request[i][1]
        v_lon = request[i][2]
        v_avg_speed = request[i][3]
        v_soc = request[i][4]

        print("\nVehicle " + str(vid) + ":\n")

        for j in range(10):
            s_id = station[j]
            dist = haversine_dist(v_lat, v_lon, s_id.lat, s_id.lon)

            if dist > v_soc:
                continue
            else:
                time = dist / v_avg_speed

                # round of time to 2 decimal places
                time = (time * 1000 + .5) / 1000.0

                print(" is " + str(time) + " hours away from Station " + str(s_id.sid) + "\n")

                v_map[vid].append([s_id.sid, s_id.coc, time])
                s_map[s_id.sid].append([vid, time])


# index is passed to sort the reachable stations according to index
def wait_count(index):
    # explore the vehicle map and waiting time
    for key in v_map:
        v_id = key
        reach = v_map[key]

        # A list to contain all possible station having minimum equal coc and wait queue time
        solver = [[]]
        solver.clear()

        for i in range(len(reach)):
            s = reach[i][0]
            time = reach[i][2]

            s_id = station[s - 1]

            wait_time = 0
            if len(s_id.queue) > 0:
                wait_time = s_id.queue[-1][1]

            solver.append([s_id, time, s_id.coc, wait_time])

        # sort the vector solver by wait time
        solver.sort(key=lambda x: x[index])

        # print the affordable station to go

        print("\nVehicle " + str(v_id) + " can recharge its battery at Station :\n")

        size = len(solver)
        for i in range(size):
            temp = solver[i]
            print("\t\t" + str(temp[0].sid) + " in " + str(temp[1]), end=" ")
            print("hours, where cost of per unit charge is " + str(temp[2]) + " and wait time is: " + str(temp[3]))

        # insert reserved vehicle vid in waiting queue at station sid.id
        s_id = solver[0][0]
        wt = request[v_id - 1][4] / s_id.roc  # time to charge vehicle vid at station sid

        if len(s_id.queue) > 0:
            wt += s_id.queue[-1][1]

        s_id.queue.append([v_id, wt])

    # array to store waiting queue length at each station
    wait = []

    for i in range(10):
        s_id = station[i]

        if len(s_id.queue):
            wait.append(len(s_id.queue) - 1)  # -1, since first vehicle is not waiting
        else:
            wait.append(0)

    print(wait)

    # dataset is wait list
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # there is only 10 stations

    fig = plt.figure(figsize=(10, 5))

    # creating the bar plot
    plt.bar(s, wait, color='maroon', width=0.4)

    plt.xlabel("Station ID")
    plt.ylabel("Waiting Queue length")
    plt.title("Number of Vehicles Waiting at different Stations")
    plt.show()


def clear_wait_queue(st):

    for s_name in st:
        s_name.queue.clear()


if __name__ == "__main__":

    # creating station specification
    s1 = Station(1, 19.2514, 72.8519, 131.0, 8.5)
    s2 = Station(2, 19.1330, 73.2297, 159.0, 8.6)
    s3 = Station(3, 19.2016, 73.1202, 133.0, 8.5)
    s4 = Station(4, 19.3760, 72.8777, 143.0, 9)
    s5 = Station(5, 19.4013, 72.5483, 105.0, 9.3)
    s6 = Station(6, 19.1179, 72.7631, 175.0, 9.2)
    s7 = Station(7, 19.2364, 72.5296, 153.0, 10)
    s8 = Station(8, 19.0596, 72.5295, 106.0, 9.7)
    s9 = Station(9, 19.0186, 73.1174, 114.0, 10.1)
    s10 = Station(10, 19.3075, 72.6263, 116.0, 9.3)

    station = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]

    # vehicles recharge request [vehicle id, latitude, longitude, avg speed, soc]

    request = [[1, 17.59, 73.98, 70, 220], [2, 17.88, 71.8, 72, 181], [3, 17.8, 71.89, 65, 174],
               [4, 20.5, 73.8, 67, 162], [5, 18.7, 71.9, 50, 87], [6, 18.7, 73.9, 60, 95],
               [7, 18.7, 74.02, 70, 110], [8, 19.3, 70.7, 47, 200], [9, 17.58, 73.9, 70, 210],
               [10, 17.88, 72.18, 72, 181], [11, 17.78, 71.9, 65, 174], [12, 20.45, 73, 67, 162],
               [13, 18.67, 71.9, 50, 87], [14, 18.667, 73.9, 60, 95], [15, 18.7, 74.03, 70, 100],
               [16, 19.28, 70.7, 47, 200], [17, 17.5, 73.98, 70, 220], [18, 17.78, 71.8, 72, 181],
               [19, 17.88, 71.89, 65, 174], [20, 20.3, 73.8, 67, 162], [21, 18.67, 71.9, 50, 87],
               [22, 18.7, 73.92, 60, 95], [23, 18.71, 74.02, 70, 110], [24, 19.31, 70.7, 47, 200],
               [25, 17.58, 73.91, 70, 210], [26, 17.88, 73.28, 72, 181], [27, 17.8, 71.9, 65, 174],
               [28, 20.46, 73, 67, 162], [29, 18.16, 71.9, 50, 187], [30, 18.717, 73.89, 60, 95],
               [31, 18.71, 74.03, 70, 100], [32, 19.3, 70.7, 47, 200], [33, 19.32, 70.7, 47, 205],
               [34, 17.5, 73.98, 70, 220], [35, 17.88, 71.81, 72, 181], [36, 17.8, 71.891, 65, 170],
               [37, 20.51, 73.8, 67, 160], [38, 18.72, 71.9, 50, 89], [39, 18.72, 73.9, 60, 90],
               [40, 18.72, 74.12, 70, 110], [41, 19.32, 70.7, 47, 210], [42, 17.6, 73.9, 70, 210],
               [43, 17.8, 71.818, 72, 180], [44, 17.7, 71.9, 65, 175], [45, 20.465, 73, 67, 160],
               [46, 18.613, 71.9, 50, 80], [47, 18.617, 73.92, 60, 95], [48, 18.71, 74.03, 70, 150],
               [49, 19.38, 70.7, 47, 200], [50, 17.52, 73.98, 70, 200], [51, 17.78, 71.83, 72, 186],
               [52, 17.88, 71.85, 65, 170], [53, 20.33, 73.8, 67, 168], [54, 18.73, 71.9, 50, 97],
               [55, 18.71, 73.92, 60, 105], [56, 18.71, 74.12, 70, 120], [57, 19.35, 70.7, 47, 200],
               [58, 17.58, 73.92, 70, 200], [59, 17.88, 71.618, 72, 180], [60, 17.82, 71.91, 65, 174],
               [61, 20.426, 73, 67, 162], [62, 18.716, 71.912, 50, 89], [63, 18.7127, 73.89, 60, 95],
               [64, 18.712, 74.03, 70, 100], [65, 19.33, 70.7, 47, 200], [66, 19.321, 70.7, 47, 205],
               [67, 17.591, 73.98, 70, 220], [68, 17.539, 73.98, 70, 220], [69, 17.88, 71.778, 72, 181],
               [70, 17.7568, 71.821, 72, 181], [71, 17.88, 71.82, 72, 181], [72, 17.8, 71.9, 65, 174],
               [73, 20.54, 73.8, 67, 162], [74, 18.7, 71.912, 50, 87], [75, 18.712, 73.9, 60, 95],
               [76, 18.714, 74.02, 70, 110], [77, 19.3, 70.712, 47, 200], [78, 17.585, 73.9, 70, 210],
               [79, 17.88, 71.7178, 72, 181], [80, 17.78, 71.912, 65, 174], [81, 20.45, 73.13, 67, 150],
               [82, 18.67127, 71.91, 50, 87], [83, 18.717, 73.945, 60, 95], [84, 18.765, 74.03, 70, 100],
               [85, 19.278, 70.7, 47, 200], [86, 17.55, 73.98, 70, 220], [87, 17.78, 71.823, 72, 181],
               [88, 17.865, 71.89, 65, 174], [89, 20.36, 73.8, 67, 162], [90, 18.67, 71.95, 50, 87],
               [91, 18.723, 73.92, 60, 95], [92, 18.71, 74.022, 70, 110], [93, 19.315, 70.7, 47, 200],
               [94, 17.58, 73.913, 70, 210], [95, 17.88, 71.8238, 72, 181], [96, 17.8, 71.936, 65, 174],
               [97, 20.46, 73.5, 67, 162], [98, 18.7167, 71.9, 50, 87], [99, 18.617, 73.879, 60, 95],
               [100, 18.71, 74.043, 70, 100], [101, 19.32, 70.71, 47, 210], [102, 19.32, 70.723, 47, 205],
               [103, 17.595, 73.98, 70, 220], [104, 17.88, 71.823, 72, 181], [105, 17.878, 71.89, 65, 174],
               [106, 20.5, 73.856, 67, 162], [107, 18.7, 71.815, 50, 87], [108, 18.7, 73.876, 60, 95],
               [109, 18.687, 74.02, 70, 110], [110, 19.287, 70.7, 47, 200], [111, 17.58, 73.911, 70, 210],
               [112, 17.868, 71.8018, 72, 181], [113, 17.78, 71.901, 65, 174], [114, 20.455, 73.1, 67, 162],
               [115, 18.66167, 71.9, 50, 87], [116, 18.7717, 73.789, 60, 95], [117, 18.7, 74.0113, 70, 170],
               [118, 19.28, 70.687, 47, 200], [119, 17.511, 73.98, 70, 220], [120, 17.78, 71.8012, 72, 181],
               [121, 17.856, 71.89, 65, 174], [122, 20.3, 73.8012, 67, 162], [123, 18.675, 71.9, 50, 87],
               [124, 18.7012, 73.92, 60, 95], [125, 18.71, 74.023, 70, 110], [126, 19.31, 70.712, 47, 200],
               [127, 17.58, 73.871, 70, 210], [128, 17.88, 71.8278, 72, 181], [129, 17.81, 71.911, 65, 174],
               [130, 20.461, 73.12, 67, 162], [131, 18.716, 71.9013, 50, 87], [132, 18.617, 73.8669, 60, 95],
               [133, 18.71, 74.0123, 70, 140], [134, 19.3, 70.741, 47, 200], [135, 19.32, 70.723, 47, 205],
               [136, 20.444, 73.8, 67, 162], [137, 18.7, 71.769, 50, 157], [138, 18.7, 73.901, 60, 95],
               [139, 18.7, 74.0022, 70, 120], [140, 19.3, 70.667, 47, 200], [141, 17.58, 73.849, 70, 210],
               [142, 17.88, 71.71238, 72, 181], [143, 17.765, 71.9, 65, 174], [144, 20.45, 73.001, 67, 162],
               [145, 18.6717, 71.569, 50, 87], [146, 18.75170, 73.9, 60, 95], [147, 18.7, 74.032, 70, 100],
               [148, 19.2128, 70.7, 47, 200], [149, 17.502, 73.918, 70, 220], [150, 17.78, 71.8018, 72, 181]
               ]

    for i in range(6):
        s_id = station[i]
        s_id.queue.append([(i+21), 2])     # append {vid, charging time} in waiting queue

    # find vehicle reach and wait list and draw graph
    for x in range(3):
        if x == 0:
            # find reachable station for vehicle
            reachable_station(request[:50])
        elif x == 1:
            reachable_station(request[50:100])
        else:
            reachable_station(request[100:])

        # # reserve station according to minimum cost
        # reserve()

        # waiting queue length count
        wait_count(3)    # 3 because in reachable station at index 3 wait time is stored

    # print(len(s3.queue))

    # clear the waiting queue
    clear_wait_queue(station)

    # measure the size of waiting queue when decision is based on distance only
    for x in range(3):
        if x == 0:
            # find reachable station for vehicle
            reachable_station(request[:50])
        elif x == 1:
            reachable_station(request[50:100])
        else:
            reachable_station(request[100:])

        # waiting queue length count
        wait_count(1)    # 1 because in reachable station at index 1 time = dist/avg_speed is stored

    # clear the waiting queue
    clear_wait_queue(station)

    # measure the size of waiting queue when decision is based on minimum cost only
    for x in range(3):
        if x == 0:
            # find reachable station for vehicle
            reachable_station(request[:50])
        elif x == 1:
            reachable_station(request[50:100])
        else:
            reachable_station(request[100:])

        # # reserve station according to minimum cost
        # reserve_according_to_cost()

        # waiting queue length count
        wait_count(2)    # 2 because in reachable station at index 2 cost of per unit charge is stored









