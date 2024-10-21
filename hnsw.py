#!python3
import sys
import numpy as np
import time
import random
from math import log2
from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
from operator import itemgetter


def l2_distance(a, b):
    return np.linalg.norm(a - b)


def heuristic(candidates, curr, k, distance_func, data):
    candidates = sorted(candidates, key=lambda a: a[1])
    result_indx_set = {candidates[0][0]}
    result = [candidates[0]]
    added_data = [data[candidates[0][0]]]
    for c, curr_dist in candidates[1:]:
        c_data = data[c]
        if curr_dist < min(map(lambda a: distance_func(c_data, a), added_data)):
            result.append((c, curr_dist))
            result_indx_set.add(c)
            added_data.append(c_data)
    for (
        c,
        curr_dist,
    ) in (
        candidates
    ):  # optional. uncomment to build neighborhood exactly with k elements.
        if len(result) < k and (c not in result_indx_set):
            result.append((c, curr_dist))

    return result


def k_closest(candidates: list, curr, k, distance_func, data):
    return sorted(candidates, key=lambda a: a[1])[:k]


class HNSW:

    def _distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def vectorized_distance_(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def __init__(
        self,
        distance_func,
        m=5,
        ef=200,
        ef_construction=30,
        m0=None,
        neighborhood_construction=heuristic,
        vectorized=False,
    ):
        self.data = []
        self.distance_func = distance_func

        if vectorized:
            self.distance = self._distance
            self.vectorized_distance = distance_func
        else:
            self.distance = distance_func
            self.vectorized_distance = self.vectorized_distance_

        self._m = m
        self._ef = ef
        self._ef_construction = ef_construction
        self._m0 = 2 * m if m0 is None else m0
        self._level_mult = 1 / log2(m)
        self._graphs = []
        self._enter_point = None

    def add(self, elem, ef=None):

        if ef is None:
            ef = self._ef

        distance = self.distance
        data = self.data
        graphs = self._graphs
        point = self._enter_point
        m = self._m

        level = int(-log2(random.random()) * self._level_mult) + 1

        idx = len(data)
        data.append(elem)

        if point is not None:
            dist = distance(elem, data[point])
            for layer in reversed(graphs[level:]):
                point, dist = self._search_layer1(elem, point, dist, layer)

            ep = [(-dist, point)]
            for layer_level, layer in enumerate(reversed(graphs[:level])):
                max_neighbors = m if layer_level != 0 else self._m0
                ep = self._search_layer2(elem, ep, layer, ef)
                layer[idx] = {}
                self._select(layer[idx], ep, max_neighbors, layer, heap=True)
                for neighbor_idx, dist in layer[idx].items():
                    self._select(layer[neighbor_idx], (idx, dist), max_neighbors, layer)

        for _ in range(len(self._graphs), level):
            self._graphs.append({idx: {}})
            self._enter_point = idx

    def search(self, q, k=1, ef=10, level=0, return_observed=True):
        graphs = self._graphs
        point = self._enter_point
        for layer in reversed(graphs[level:]):
            point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]

        return self.beam_search(
            graph=graphs[level],
            q=q,
            k=k,
            eps=[point],
            ef=ef,
            return_observed=return_observed,
        )

    def beam_search(
        self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False
    ):
        """
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        """
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = (
            dict()
        )  # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color="red", marker="^")
            ax.annotate("query", (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(
                    x=self.data[current_vertex][0],
                    y=self.data[current_vertex][1],
                    s=marker_size,
                    color="red",
                )
                ax.annotate(len(visited), self.data[current_vertex])

            # check stop conditions #####
            observed_sorted = sorted(observed.items(), key=lambda a: a[1])
            # print(observed_sorted)
            ef_largets = observed_sorted[min(len(observed) - 1, ef - 1)]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    if ax:
                        ax.scatter(
                            x=self.data[neighbor][0],
                            y=self.data[neighbor][1],
                            s=marker_size,
                            color="yellow",
                        )
                        ax.annotate(len(visited), self.data[neighbor])

        observed_sorted = sorted(observed.items(), key=lambda a: a[1])
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    def _search_layer1(self, q, entry_point, dist_to_entry, layer):
        visited = set()
        candidates = [(dist_to_entry, entry_point)]
        best_point = entry_point
        best_dist = dist_to_entry
        visited.add(entry_point)

        while candidates:
            dist, current = heappop(candidates)
            if dist > best_dist:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(
                q, [self.data[n] for n in neighbors]
            )

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                if neighbor_dist < best_dist:
                    best_point = neighbor
                    best_dist = neighbor_dist
                    heappush(candidates, (neighbor_dist, neighbor))

        return best_point, best_dist

    def _search_layer2(self, q, ep, layer, ef):
        visited = set()
        candidates = [(-dist, idx) for dist, idx in ep]
        heapify(candidates)
        visited.update(idx for _, idx in ep)

        while candidates:
            dist, current = heappop(candidates)
            if dist > -ep[0][0]:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(
                q, [self.data[n] for n in neighbors]
            )

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                mdist = -neighbor_dist
                if len(ep) < ef:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heappush(ep, (mdist, neighbor))
                elif mdist > ep[0][0]:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heapreplace(ep, (mdist, neighbor))

        return ep

    def _select(self, neighbors, candidates, m, layer, heap=False):
        neighbor_dicts = [layer[idx] for idx in neighbors]

        def prioritize(idx, dist):
            proximity = any(nd.get(idx, float("inf")) < dist for nd in neighbor_dicts)
            return proximity, dist, idx

        if heap:
            candidates = nsmallest(
                m, (prioritize(idx, -mdist) for mdist, idx in candidates)
            )
            unchecked = m - len(neighbors)
            candidates_to_add = candidates[:unchecked]
            candidates_to_check = candidates[unchecked:]

            if candidates_to_check:
                to_remove = nlargest(
                    len(candidates_to_check),
                    (prioritize(idx, dist) for idx, dist in neighbors.items()),
                )
            else:
                to_remove = []

            for _, dist, idx in candidates_to_add:
                neighbors[idx] = dist

            for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zip(
                candidates_to_check, to_remove
            ):
                if (p_old, d_old) <= (p_new, d_new):
                    break
                del neighbors[idx_old]
                neighbors[idx_new] = d_new
        else:
            idx, dist = candidates
            candidates = [prioritize(idx, dist)]
            if len(neighbors) < m:
                neighbors[idx] = dist
            else:
                max_idx, max_val = max(neighbors.items(), key=itemgetter(1))
                if dist < max_val:
                    del neighbors[max_idx]
                    neighbors[idx] = dist

    def __getitem__(self, idx):
        for layer in self._graphs:
            if idx in layer:
                yield from layer[idx].items()
            else:
                return
