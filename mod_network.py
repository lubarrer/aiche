import numpy as np
import networkx as nx
from numpy_financial import numpy_financial as npf


def law_of_cosines(len1, len2, angle):
    return np.sqrt(len1 ** 2 + len2 ** 2 - 2 * len1 * len2 * np.cos(np.radians(angle)))


def distance(i, j):
    return law_of_cosines(dist[i], dist[j], np.abs(((i - j) * 22.5)))


def distance_iter():
    all = []
    well_1 = []
    well_2 = []
    for i in range(len(dist)):
        for j in range(len(dist)):
            if i != j:
                k = distance(i, j)
                all.append(k)
                well_1.append(i)
                well_2.append(j)
    return all, well_1, well_2


def prod_decline(prod, years_deployed):
    if years_deployed < 20 and years_deployed >= 0:
        production = np.zeros(20)
        j = 0
        while j <= years_deployed:
            if j > 1:
                prod = prod * 0.65
                production[j] = prod
            else:
                production[j] = prod
            j += 1
        return production[years_deployed]
    else:
        print("Year value must be between 0 and 19")


def unit_size(mod_size, density, lpg, naptha, diesel, w=0):
    size = ["small", "medium", "large"]
    mod_capacity = [500, 2500, 5000]
    if mod_size in size:
        module_capacity = mod_capacity[size.index(mod_size)]

    base = density * (1.25 * lpg + 1.79 * naptha + 2.14 * diesel) * 365 * 0.8
    if w == 0:
        return module_capacity * base
    else:
        return w * base


def truck_contracts(load, distance):
    global truck_id
    global available
    global active_contract
    global yrs_til_expiration

    x = int(len(truck_id) - 1)
    y = int(load / 5000) + (load % 5000 > 0)
    for i in range(y):
        # Claim truck and take off roster
        if available[x] == True and active_contract[x] == True:
            available[x] = False

        # Renew contract if expired
        elif active_contract[x] == False or yrs_til_expiration[x] == 0:
            active_contract[x] = True
            yrs_til_expiration[x] = 5
        #            print("Contract for truck #{} has been renewed.".format(truck_id[x]))

        # Add new truck contract if no trucks available
        else:
            truck_id.append(len(truck_id) + 1)
            available = np.append(available, False)
            active_contract = np.append(active_contract, True)
            yrs_til_expiration.append(5)
    return -distance * 1.25 * y, truck_id


def redeploy_module(mod_size, distance):  # move a module from one well to another well
    if mod_size == "small":
        return -(
            500 * 3 * distance + (unit_size("small", 1, 0.33, 0.33, 0.33)) * 30 / 292
        )
    elif mod_size == "medium":
        return -(
            2500 * 3 * distance + (unit_size("medium", 1, 0.33, 0.33, 0.33)) * 30 / 292
        )
    elif mod_size == "large":
        return -(
            5000 * 3 * distance + (unit_size("large", 1, 0.33, 0.33, 0.33)) * 30 / 292
        )


def deploy_unit(
    mod_size, miles, years_deployed, prod, deduct=0
):  # deploy a module to the first well
    global gas_prod
    size = ["small", "medium", "large"]
    mod_capacity = [500, 2500, 5000]
    if mod_size in size:
        module_capacity = mod_capacity[size.index(mod_size)]
    well_capacity = prod_decline(prod, years_deployed) - deduct
    if well_capacity < module_capacity:
        revenue = (
            unit_size(mod_size, 1, 0.33, 0.33, 0.33, w=well_capacity) * 365 * 0.8
            - miles * module_capacity * 3
        )
    else:
        revenue = (
            unit_size(mod_size, 1, 0.33, 0.33, 0.33, w=0) * 365 * 0.8
            - miles * module_capacity * 3
        )
    well = gas_prod.index(prod)
    return revenue, well


def parallel_units(number_of_small, number_of_medium, number_of_large):
    global module_id
    global module_size
    global mod_available
    global capacity
    equipment_cost = (
        number_of_small * 500_000
        + number_of_medium * 2_500_000
        + number_of_large * 5_000_000
    )  # Need to update equipment pricing
    i = 0
    while i < len(range(number_of_large)):
        module_id.append("large {}".format(i + 1))
        module_size.append("large")
        mod_available.append(bool(True))
        capacity.append(5000)
        i += 1
    i = 0
    while i < len(range(number_of_medium)):
        module_id.append("medium {}".format(i + 1))
        module_size.append("medium")
        mod_available.append(bool(True))
        capacity.append(2500)
        i += 1
    i = 0
    while i < len(range(number_of_small)):
        module_id.append("small {}".format(i + 1))
        module_size.append("small")
        mod_available.append(bool(True))
        capacity.append(500)
        i += 1
    return equipment_cost


def check_redeploy(current_well, id, year, deduct=0):
    global module_id
    mod_size = module_size[module_id.index(id)]
    years_deployed = 0
    nxt_profit = 0
    revenue = 0
    max_prod = [(nx.get_node_attributes(G, "Production")[x]) for x in G]
    max_prod.sort()
    while nxt_profit <= revenue:
        revenue, _ = deploy_unit(
            mod_size,
            0,
            years_deployed,
            nx.get_node_attributes(G, "Production")[current_well],
            deduct=deduct,
        )
        nxt_well = max_prod[-2]
        i = 2
        while gas_prod.index(nxt_well) == current_well:
            nxt_well = max_prod[-i]
            i += 1
        nw = gas_prod.index(nxt_well)
        nxt_revenue, _ = deploy_unit(
            mod_size, G[current_well][nw]["weight"], 0, nxt_well
        )
        truck_cost, _ = truck_contracts(
            capacity[module_size.index(mod_size)], G[current_well][nw]["weight"]
        )
        downtime_cost = redeploy_module(mod_size, G[current_well][nw]["weight"])
        nxt_profit = nxt_revenue + truck_cost + downtime_cost
        years_deployed += 1
    if nxt_profit >= revenue and year == years_deployed - 1:
        truck_cost, _ = truck_contracts(
            capacity[module_size.index(mod_size)], G[current_well][nw]["weight"]
        )
        downtime_cost = redeploy_module(mod_size, G[current_well][nw]["weight"])
        redeployment_cost = truck_cost + downtime_cost
        #        print(
        #            "module {} redeployed to well {}\nShipment cost ${:,.2f} Downtime cost ${:,.2f}.".format(
        #                id, nx.get_node_attributes(G, "Well")[nw], truck_cost, downtime_cost
        #            )
        #        )
        return redeployment_cost, nw
    else:
        return 0, current_well


def init_deploy():
    global annual_profit
    i = 1
    current_wells = []
    deduct = 0
    for id in module_id:
        size = module_size[module_id.index(id)]
        max_prod = [(nx.get_node_attributes(G, "Production")[x]) for x in G]
        max_prod.sort()
        _, current_well = deploy_unit(size, 0, 0, max_prod[-i])
        if current_well in current_wells:
            deduct += capacity[module_id.index(id)]
            revenue, current_well = deploy_unit(size, 0, 0, max_prod[-i], deduct=deduct)
            revenue2, _ = deploy_unit(size, 0, 0, max_prod[-i - 1])
            if revenue < revenue2:
                revenue, current_well = deploy_unit(size, 0, 0, max_prod[-i - 1])
                deduct = 0
                i += 1
            #            print(
            #                "Module {} initial deployment at Well {}".format(id, well[current_well])
            #            )
            current_wells.append(current_well)
        else:
            revenue, current_well = deploy_unit(size, 0, 0, max_prod[-i])
            #            print(
            #                "Module {} initial deployment at Well {}".format(id, well[current_well])
            #            )
            current_wells.append(current_well)
        annual_profit[0] += revenue
    return current_wells


well = [
    "1A",
    "2A",
    "1B",
    "2B",
    "1C",
    "2C",
    "1D",
    "2D",
    "1E",
    "2E",
    "1F",
    "2F",
    "1G",
    "2G",
    "1H",
    "2H",
]
gas_prod = [
    4494,
    9182,
    2989,
    12258,
    8365,
    8742,
    7447,
    3840,
    3290,
    7939,
    14737,
    2874,
    11155,
    6076,
    7081,
    6468,
]
unused_prod = [
    4494,
    9182,
    2989,
    12258,
    8365,
    8742,
    7447,
    3840,
    3290,
    7939,
    14737,
    2874,
    11155,
    6076,
    7081,
    6468,
]
dist = [
    62.0,
    85.8,
    21.0,
    84.7,
    17.0,
    2.4,
    73.4,
    42.0,
    6.2,
    98.8,
    91.0,
    13.0,
    59.0,
    87.0,
    89.3,
    36.0,
]
deployed_module = np.array(
    [
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]
)

attr = {
    0: {"Production": gas_prod[0], "Well": well[0]},
    1: {"Production": gas_prod[1], "Well": well[1]},
    2: {"Production": gas_prod[2], "Well": well[2]},
    3: {"Production": gas_prod[3], "Well": well[3]},
    4: {"Production": gas_prod[4], "Well": well[4]},
    5: {"Production": gas_prod[5], "Well": well[5]},
    6: {"Production": gas_prod[6], "Well": well[6]},
    7: {"Production": gas_prod[7], "Well": well[7]},
    8: {"Production": gas_prod[8], "Well": well[8]},
    9: {"Production": gas_prod[9], "Well": well[9]},
    10: {"Production": gas_prod[10], "Well": well[10]},
    11: {"Production": gas_prod[11], "Well": well[11]},
    12: {"Production": gas_prod[12], "Well": well[12]},
    13: {"Production": gas_prod[13], "Well": well[13]},
    14: {"Production": gas_prod[14], "Well": well[14]},
    15: {"Production": gas_prod[15], "Well": well[15]},
}

G = nx.Graph()
H = nx.complete_graph(16)
G.add_weighted_edges_from((u, v, distance(u, v)) for u, v in H.edges())
nx.set_node_attributes(G, attr)
nx.write_gpickle(G, "pathToGraphPickleFile.nx")

truck_id = [1]
available = [False]
active_contract = [False]
yrs_til_expiration = [0]

module_id = []
module_size = []
mod_available = []
capacity = []

num_small = 0
num_medium = 0
num_large = 1

equipment_cost = parallel_units(num_small, num_medium, num_large)

year = range(0, 20)
annual_profit = np.zeros(20)
occupied_wells = init_deploy()
Gs = []
for i in range(len(occupied_wells)):
    Gs.append(nx.read_gpickle("pathToGraphPickleFile.nx"))

years_deployed_list = np.ones(len(occupied_wells))
current_wells = np.zeros(len(occupied_wells), dtype=int)
new_wells = np.zeros(len(occupied_wells), dtype=int)
revenues = np.zeros(len(occupied_wells))
for y in year[1:]:
    # print("Year {}".format(y))
    for id in module_id:
        mod_iter = module_id.index(id)
        years_deployed = int(years_deployed_list[mod_iter])
        size = module_size[mod_iter]
        G = Gs[mod_iter]
        deduct = 0
        if current_wells[mod_iter] != 0:
            count = np.count_nonzero(np.array(capacity) == capacity[0])
            deduct += capacity[mod_iter] * count
        if y == 1:
            current_wells[mod_iter] = occupied_wells[mod_iter]
            redeploy_cost, new_wells[mod_iter] = check_redeploy(
                current_wells[mod_iter], id, years_deployed
            )
        else:

            redeploy_cost, new_wells[mod_iter] = check_redeploy(
                current_wells[mod_iter], id, years_deployed
            )

        if new_wells[mod_iter] == current_wells[mod_iter]:
            revenues[mod_iter], current_wells[mod_iter] = deploy_unit(
                size, 0, years_deployed, gas_prod[new_wells[mod_iter]], deduct=deduct
            )
            annual_profit[y] += revenues[mod_iter]
            years_deployed_list[mod_iter] += 1

        else:
            revenues[mod_iter], _ = deploy_unit(
                size,
                G[current_wells[mod_iter]][new_wells[mod_iter]]["weight"],
                years_deployed,
                gas_prod[new_wells[mod_iter]],
            )
            count = np.count_nonzero(current_wells == current_wells[mod_iter])
            G.remove_node(current_wells[mod_iter])
            _, current_wells[mod_iter] = deploy_unit(
                size, 0, years_deployed, gas_prod[new_wells[mod_iter]]
            )
            annual_profit[y] += revenues[mod_iter] - redeploy_cost
            years_deployed_list[mod_iter] = 0

npv = npf.npv(0.03, annual_profit)
print("Net Present Value = ${:,.2f}".format(npv))
