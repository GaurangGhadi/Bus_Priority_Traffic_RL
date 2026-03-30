# bus_priority.py
# gym wrapper for sumo-rl that adds bus priority to signal control.
# buses carry ~40 people vs ~1.5 per car, so we weight delays by
# occupancy and the agent learns to prioritize them.

import numpy as np
import gymnasium as gym
import xml.etree.ElementTree as ET
import os


class BusPriorityWrapper(gym.Wrapper):

    def __init__(self, env, beta=2.0, avg_bus_passengers=40, car_occupancy=1.5):
        super().__init__(env)
        self.beta = beta
        self.avg_bus_passengers = avg_bus_passengers
        self.car_occupancy = car_occupancy
        self.ts_id = list(env.traffic_signals.keys())[0]
        self.lanes = env.traffic_signals[self.ts_id].lanes
        self.num_lanes = len(self.lanes)

        # augmented obs: original + bus count per lane + total bus wait + max bus wait
        orig = env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(orig + self.num_lanes + 2,), dtype=np.float32)

        self.prev_person_delay = 0.0
        self.buses_seen = 0
        self.cars_seen = 0
        self.steps = 0

    def _is_bus(self, veh_id):
        # check type ID, not vClass. setting vClass="bus" in the vType
        # makes sumo silently drop them since the lanes are passenger-only.
        try:
            sumo = self.env.traffic_signals[self.ts_id].sumo
            return sumo.vehicle.getTypeID(veh_id) == "bus"
        except:
            return False

    def step(self, action):
        obs, original_reward, done, truncated, info = self.env.step(action)
        self.steps += 1
        sumo = self.env.traffic_signals[self.ts_id].sumo

        # collect bus features and person delay in one pass over all lanes
        bus_counts = []
        bus_waits = []
        car_wait, bus_wait = 0.0, 0.0

        for lane in self.lanes:
            count = 0
            for veh in sumo.lane.getLastStepVehicleIDs(lane):
                wait = sumo.vehicle.getWaitingTime(veh)
                if self._is_bus(veh):
                    count += 1
                    bus_waits.append(wait)
                    bus_wait += wait
                    self.buses_seen += 1
                else:
                    car_wait += wait
                    self.cars_seen += 1
            bus_counts.append(count)

        total_bus_wait = sum(bus_waits) if bus_waits else 0.0
        max_bus_wait = max(bus_waits) if bus_waits else 0.0
        bus_features = np.array(bus_counts + [total_bus_wait, max_bus_wait], dtype=np.float32)
        augmented_obs = np.concatenate([obs, bus_features])

        # person_delay = car_wait * 1.5 + beta * bus_wait * 40
        person_delay = car_wait * self.car_occupancy + self.beta * bus_wait * self.avg_bus_passengers

        # reward = change in person delay, scaled down
        # R(t) = (person_delay(t-1) - person_delay(t)) / 100
        reward = (self.prev_person_delay - person_delay) / 100.0
        self.prev_person_delay = person_delay

        info["bus_reward"] = reward
        info["original_reward"] = original_reward
        info["car_wait"] = car_wait
        info["bus_wait"] = bus_wait
        info["person_delay"] = person_delay
        info["buses_this_step"] = sum(bus_counts)

        return augmented_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_person_delay = 0.0
        self.buses_seen = 0
        self.cars_seen = 0
        self.steps = 0
        zeros = np.zeros(self.num_lanes + 2, dtype=np.float32)
        return np.concatenate([obs, zeros]), info

    def get_debug_info(self):
        return {
            "total_buses_seen": self.buses_seen,
            "total_cars_seen": self.cars_seen,
            "steps": self.steps,
            "avg_buses_per_step": self.buses_seen / max(1, self.steps),
        }


def generate_bus_routes(base_route_file, output_route_file,
                        net_file=None, bus_frequency=300, simulation_end=5000):
    """reads an existing route file and adds bus vehicle type + bus flows."""
    tree = ET.parse(base_route_file)
    root = tree.getroot()

    # add bus vType at position 0 (must come before flows that reference it).
    # do NOT set vClass="bus" — the lanes only allow passenger class and
    # sumo will silently refuse to spawn them. we detect by type ID instead.
    if not any(vt.get("id") == "bus" for vt in root.findall("vType")):
        vtype = ET.Element("vType")
        for key, val in [("id", "bus"), ("accel", "1.0"), ("decel", "3.0"),
                         ("sigma", "0.5"), ("length", "12.0"), ("minGap", "3.0"),
                         ("maxSpeed", "15.0"), ("guiShape", "bus"), ("color", "1,0.5,0")]:
            vtype.set(key, val)
        root.insert(0, vtype)

    # figure out what edges to use for bus routes
    bus_routes = _get_bus_routes(net_file, base_route_file, bus_frequency)

    for bus_line in bus_routes:
        route_id = f"route_{bus_line['id']}"
        if root.find(f".//route[@id='{route_id}']") is not None:
            continue

        route = ET.SubElement(root, "route")
        route.set("id", route_id)
        route.set("edges", bus_line["edges"])

        flow = ET.SubElement(root, "flow")
        for key, val in [("id", bus_line["id"]), ("type", "bus"), ("route", route_id),
                         ("begin", "0"), ("end", str(simulation_end)),
                         ("period", str(int(bus_line.get("period", bus_frequency)))),
                         ("departLane", "best"), ("departPos", "base"), ("departSpeed", "max")]:
            flow.set(key, val)

    tree.write(output_route_file, xml_declaration=True, encoding="UTF-8")
    print(f"wrote bus routes to {output_route_file}")
    for bus_line in bus_routes:
        print(f"  {bus_line['id']}: {bus_line['edges']}")
    return output_route_file


def _get_bus_routes(net_file, route_file, frequency):
    """auto-detect edges from .net.xml, fall back to pattern matching."""

    # try to find the net file if not given
    if net_file is None:
        guess = os.path.join(os.path.dirname(route_file), "single-intersection.net.xml")
        if os.path.exists(guess):
            net_file = guess

    # parse the network and pair incoming/outgoing edges at the junction
    if net_file and os.path.exists(net_file):
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            junctions = [j for j in root.findall("junction")
                         if j.get("type") not in ("internal", "dead_end")]
            if junctions:
                junction_id = junctions[0].get("id")
                incoming, outgoing = [], []
                for edge in root.findall("edge"):
                    edge_id = edge.get("id")
                    if edge_id.startswith(":"):
                        continue
                    if edge.get("to") == junction_id:
                        incoming.append((edge_id, edge.get("from")))
                    elif edge.get("from") == junction_id:
                        outgoing.append((edge_id, edge.get("to")))

                lines, used = [], set()
                for in_edge, in_node in incoming:
                    for out_edge, out_node in outgoing:
                        if out_node == in_node or (in_node, out_node) in used:
                            continue
                        used.add((in_node, out_node))
                        lines.append({"id": f"bus_{in_node}_to_{out_node}",
                                      "edges": f"{in_edge} {out_edge}", "period": frequency})
                if lines:
                    return lines[:4]
        except Exception as e:
            print(f"  net parse failed ({e}), trying fallback")

    # fallback: look at edges in the route file and try known naming patterns
    tree = ET.parse(route_file)
    root = tree.getroot()
    edges = set()
    for r in root.findall("route"):
        edges.update(r.get("edges", "").split())
    for v in root.findall("vehicle"):
        r = v.find("route")
        if r is not None:
            edges.update(r.get("edges", "").split())

    patterns = [
        [("n_t", "t_s", "bus_NS"), ("s_t", "t_n", "bus_SN"),
         ("e_t", "t_w", "bus_EW"), ("w_t", "t_e", "bus_WE")],
        [("1to2", "2to5", "bus_NS"), ("5to2", "2to1", "bus_SN"),
         ("3to2", "2to4", "bus_EW"), ("4to2", "2to3", "bus_WE")],
    ]
    for pattern in patterns:
        lines = [{"id": name, "edges": f"{src} {dst}", "period": frequency}
                 for src, dst, name in pattern if src in edges and dst in edges]
        if lines:
            return lines

    return []


if __name__ == "__main__":
    import sumo_rl
    pkg = os.path.dirname(sumo_rl.__file__)
    net = os.path.join(pkg, "nets", "2way-single-intersection", "single-intersection.net.xml")
    rou = os.path.join(pkg, "nets", "2way-single-intersection", "single-intersection-vhvh.rou.xml")
    if not os.path.exists(rou):
        rou = rou.replace("-vhvh", "")
    if os.path.exists(net) and os.path.exists(rou):
        generate_bus_routes(rou, rou.replace(".rou.xml", "-with-buses.rou.xml"), net_file=net)
    else:
        print(f"can't find: {net} / {rou}")
