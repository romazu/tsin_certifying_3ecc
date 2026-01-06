import dataclasses
import itertools
import os.path
from collections import Counter
from typing import List, Dict, Set, Iterable, Callable, Union


@dataclasses.dataclass(frozen=True)
class Edge:
    start: int
    end: int
    idx: int  # not necessary: used only for parallel edges tracking
    is_virtual: bool = False  # not necessary: used only for edge label tracking and rendering

    def inversed(self):
        return Edge(self.end, self.start, self.idx, self.is_virtual)


@dataclasses.dataclass(frozen=True)
class Ear:
    backedge: Edge


@dataclasses.dataclass
class Path:
    vertices: List[int]


class ConSeq:
    def __init__(self, paths: List[Ear | Path] = None, ih: int = 0):
        self.paths = [] if paths is None else paths
        self.ih = ih

    def prepend(self, cs: Union['ConSeq', Ear, Path]):
        if isinstance(cs, ConSeq):
            self.paths = [*cs.paths, *self.paths]
        elif isinstance(cs, Ear | Path):
            self.paths = [cs, *self.paths]

    def append(self, cs: Union['ConSeq', Ear, Path]):
        if isinstance(cs, ConSeq):
            self.paths = [*self.paths, *cs.paths]
        elif isinstance(cs, Ear | Path):
            self.paths = [*self.paths, cs]


class Solver:
    def __init__(self, graph: Dict[int, List[int]], num_vertices: int):

        self.graph = graph
        self.num_vertices = num_vertices

        vertices = list(range(1, num_vertices + 1))  # 1-based
        self.parent: Dict[int, int | None] = {v: None for v in vertices}
        self.dfs = {v: 0 for v in vertices}
        self.desc_count = {v: 1 for v in vertices}  # Number of descendants including self
        # Note: lowpt is unused.
        # self.lowpt = {v: vertices[-1] + 1 for v in vertices}  # "infinity"
        self.p_hat: Dict[int, Ear | None] = {v: None for v in vertices}
        self.p_anchor: Dict[int, Ear | None] = {v: None for v in vertices}
        self.cs: Dict[int, ConSeq] = {v: ConSeq() for v in vertices}
        self.sigma = {v: [v] for v in vertices}
        self.inc: Dict[int, Set[Edge]] = {v: set() for v in vertices}

        # Note, here w and u paths are represented by actual lists instead of linked lists as in the paper.
        self.p: Dict[int, List[int]] = {v: [v] for v in vertices}

        self.ear: Dict[int, Ear | None] = {}  # Maps vertex v to ear on tree edge [parent(v), v]

        self.tchain: Dict[int, List[int]] = {}
        self.bchain: Dict[int, List[int]] = {}
        self.bchain_t: List[int] = []  # It's a self field only for tracing. Otherwise, it jsut a tmp variable.
        self.cycle: Dict[int, List[List[int]]] = {}

        self.cnt = 1
        self.bridges: List[Edge] = []
        self.components: List[int] = []

        # Not necessary.
        self.virtual_edges: List[Edge] = []

        # Only for tracking.
        self.trace_current_path: list = []
        self.trace_major_step: str | None = None
        self.trace_absorbed: Set[int] = set()
        self.trace_original_parent_patch: Dict[int, int | None] = {}
        self.trace_mader: Dict[int, List[List[int]]] = {}
        self.parallel_next_idx = Counter()
        self.parallel_in_next_idx = Counter()
        self.parallel_virtual_next_idx = Counter()

    def certifying_3_edge_connectivity(self, start_vertex: int):
        self.edge_connect_cs(start_vertex, None)

        # NOTE: New wrt to the paper.
        # There can be a bchain still attached to start_vertex, which can now be converted to a cycle.
        if self.bchain[start_vertex]:
            self.cycle[start_vertex].append([*self.bchain[start_vertex]])  # Mod: implicit attachment vertex
            self.bchain[start_vertex].clear()

        # Mod: implicit attachment vertex
        # for cycle in self.cycle[start_vertex]:
        #     # make r the starting and ending node of the cycle; note: σ(r) = σ(x)
        #     if cycle[0] != start_vertex:
        #         cycle[0] = start_vertex
        #         cycle[-1] = start_vertex

        self.cs[start_vertex].prepend(self.p_hat[start_vertex])
        self.components.append(start_vertex)
        self.trace_mader[start_vertex] = self.materialize_component_cs_paths(start_vertex, extract_core=True)
        self.trace_major_step = build_step("finish")

    def edge_connect_cs(self, w, v):
        # initialization
        self.dfs[w] = self.cnt
        self.cnt += 1
        self.parent[w] = v
        # self.lowpt[w] = self.dfs[w]
        self.tchain[w] = []
        self.bchain[w] = []
        self.bchain_t = []
        self.cycle[w] = []

        self.ear[w] = None

        self.trace_current_path.append(w)
        self.trace_major_step = build_step("enter", w=w, v=v, idx=0)

        is_parent_seen = False
        for u in self.graph[w]:
            # pick the next vertex u in the adjacency list of w
            if u == v and not is_parent_seen:
                is_parent_seen = True
                continue

            dkey = tuple(sorted([w, u]))
            if self.dfs[u] == 0:
                # u is unvisited

                idx = self.parallel_next_idx[dkey]
                self.parallel_next_idx[dkey] += 1
                self.parallel_in_next_idx[dkey] += 1  # skip tree edge index for incoming
                assert idx == 0  # tree edge is always first to encounter

                self.edge_connect_cs(u, w)

                self.desc_count[w] += self.desc_count[u]  # Accumulate descendant count for fast ancestor checks.

                if self.p_anchor[u] is None or self.s(self.p_anchor[u]) == u:
                    # equivalent to (deg(u) ≤ 2)
                    # eject super-vertex u from Pu ; finalize CS_δ(u)
                    # Note, we pass parallel edge index
                    self.gen_cs(w, u, idx, self.p[u], self.bchain_t)  # maybe overwrites bchain_t

                if self.lexlt(self.p_hat[w], self.p_hat[u]):
                    # equivalent to (lowpt(w) ≤ lowpt(u))
                    # absorb the entire u-path
                    self.trace_major_step = build_step("exit_absorb_u_start", u=u, w=w, idx=idx)
                    self.absorb_ear(w, self.p_hat[u], [w] + self.p[u], self.bchain_t)  # clears bchain_t
                    self.trace_major_step = build_step("exit_absorb_u_finish", u=u, w=w, idx=idx)
                else:
                    # self.p_hat[u] < self.p_hat[w] lexicographically smaller
                    # equivalent to (lowpt(w) > lowpt(u))
                    # absorb the entire w-path
                    self.trace_major_step = build_step("exit_absorb_w_start", u=u, w=w, idx=idx)
                    self.absorb_ear(w, self.p_hat[w], self.p[w], [])
                    # transfer bchaint to w.bchain
                    self.bchain[w] = [*self.bchain_t]  # copy
                    self.bchain_t.clear()
                    # correspond to (lowpt(w) := lowpt(u))
                    self.p[w] = [w] + self.p[u]
                    self.p_hat[w] = self.p_hat[u]
                    self.ear[w] = self.ear[u]
                    self.trace_major_step = build_step("exit_absorb_w_finish", u=u, w=w, idx=idx)

            elif self.dfs[u] < self.dfs[w]:
                # (u <- w) is an outgoing back-edge of w
                # First seen parent was already dismissed, so it's either u != v or a nontrivial parallel edge to the parent.

                idx = self.parallel_next_idx[dkey]
                self.parallel_next_idx[dkey] += 1

                e = Edge(w, u, idx)

                if self.lexlt(e, self.p_hat[w]):
                    # equivalent to dfs(u) < lowpt(w)
                    # Absorb w-path.
                    self.trace_major_step = build_step("outgoing_absorb_w_start", u=u, w=w, idx=idx)
                    self.absorb_ear(w, self.p_hat[w], self.p[w], [])
                    # correspond to (lowpt(w) := dfs(u))
                    self.p[w] = [w]
                    self.p_hat[w] = Ear(e)
                    self.ear[w] = Ear(e)
                    self.trace_major_step = build_step("outgoing_absorb_w_finish", u=u, w=w, idx=idx)
                else:
                    # there is no u-path; no chain will be updated
                    self.trace_major_step = build_step("outgoing_absorb_no_u_start", u=u, w=w, idx=idx)
                    self.absorb_ear(w, Ear(e), None, [])
                    self.trace_major_step = build_step("outgoing_absorb_no_u_finish", u=u, w=w, idx=idx)
            else:
                # (u -> w) = inversed(w -> u) is an incoming back-edge of w

                idx = self.parallel_in_next_idx[dkey]
                self.parallel_in_next_idx[dkey] += 1

                e = Edge(w, u, idx)

                # save incoming back-edge in Inc_w
                self.trace_major_step = build_step("incoming_add", u=u, w=w, idx=idx)
                self.inc[w].add(e.inversed())

        # dealing with incoming back-edges
        # NOTE: New wrt to the paper: check for >1 instead of not nil.
        if (len(self.p[w]) > 1) and len(self.inc[w]) != 0:
            # Sanity check.
            if self.p[w][0] != w:
                raise ValueError("First element of P(w) is not w")
            self.absorb_path(w, self.p[w], self.inc[w])
            self.trace_major_step = build_step("process_incoming_absorb_finish", w=w, v=v, idx=0)

        self.trace_current_path.pop()

    def gen_cs(self, w, u, idx, p_u: List[int], bchain_t: List):
        # eject super-vertex u from p_u; finalize cs[u]
        # Create a cactus node σ(u) and attach it to the corresponding tchain or bchain.

        # Mod: implicit attachment vertex. Originally start=end vertex is a part of a cycle. Now it's implicit.
        # for cycle in self.cycle[u]:
        #     # xQx represents a cycle σ(x)σ(u1)σ(u2) ...σ(uk)σ(x)
        #     # make u the starting and ending node of the cycle; note: σ(u) = σ(x)
        #     if cycle[0] != u:
        #         cycle[0] = u
        #         cycle[-1] = u

        if self.p_hat[u] is None or self.s(self.p_hat[u]) == u:
            # deg(u) = 1, i.e. (w, u) is a bridge
            # P (u) = ⊥ ⇒ σ(u) = {u}
            # Finalize CSδ(u) based on Lemma3.8(i)

            self.trace_major_step = build_step("exit_gen_cs_bridge", u=u, w=w, idx=idx)

            # Construction of a cactus representation for the 2-edge-connected component containing σ(u) is complete
            # NOTE: New wrt to the paper.
            # Actually the construction of cactus is not complete. There can be a bchain still attached to u, which can now be converted to a cycle.
            if self.bchain[u]:
                self.cycle[u].append([*self.bchain[u]])  # Mod: implicit attachment vertex
                self.bchain[u].clear()

            self.cs[u].prepend(self.p_hat[u])
            # cut off ˆP(u); Pu = nil ⇒ Pu does not exist
            self.p_hat[u] = None
            self.ear[u] = None  # bridge can not be a part of an ear
            self.p[u] = []
            # (w, u) is a bridge
            e = Edge(w, u, 0)  # bridge cannot be a parallel edge, so idx=0
            self.bridges.append(e)
        else:
            # deg(u) = 2
            if len(p_u) == 1:
                # cut-pair is {(w → u), (d ← udd)}, where (d ← udd) = ear(w → u)
                if p_u[0] != u:
                    # sanity check
                    raise ValueError("gen_cs: p_u[0] != u")

                self.trace_major_step = build_step("exit_gen_cs_2cut_back", u=u, w=w, idx=idx)

                # Pu : u, i.e. the generator of the cut-edge chain is a back-edge.
                bchain_t.clear()  # keep list reference intact
                bchain_t.extend([*self.bchain[u], u])
                # Not necessary, but we clean up absorbed chains. Intuitively, we transfer u.bchain to bchain_t.
                self.bchain[u].clear()

                # determine udd and d
                udd = self.ear[u].backedge.start
                d = self.ear[u].backedge.end

                # Note, equivalent:
                # udd = self.p_hat[u].backedge.start
                # d = self.p_hat[u].backedge.end
                if w != d:
                    # replace (d-Pear(w→u)-w) path with virtual edge (d ← w)
                    v_key = tuple(sorted([w, d]))
                    v_idx = self.parallel_virtual_next_idx[v_key]
                    self.parallel_virtual_next_idx[v_key] += 1

                    e_virtual = Edge(w, d, v_idx, is_virtual=True)
                    self.virtual_edges.append(e_virtual)
                    self.trace_major_step = build_step(
                        "new_virtual_backedge",
                        u=u, w=w, idx=idx,
                        old=self.ear[u].backedge, new=e_virtual)

                    self.p_hat[u] = Ear(e_virtual)
                    self.ear[u] = Ear(e_virtual)  # NOTE: New wrt to the paper. This is actually a new backedge P'_5.
                else:
                    # cut off the self-loop p_hat(u)
                    self.p_hat[u] = None

                    # NOTE: New wrt to the paper. It's not necessary to remove the ear.
                    # self.ear[u] = None  #
                self.p[u] = []
            else:
                # cut pair is {(w → u), (udd → u1)}, where udd = parent(u1)

                self.trace_major_step = build_step("exit_gen_cs_2cut_tree", u=u, w=w, idx=idx)

                # Pu : uu1...uℓ, ℓ ≥ 1, i.e. the generator of the cut-edge chain is a tree-edge
                # extend u1.tchain to include u, where {(u1 → u), (u → w)} is the cut-pair
                self.tchain[p_u[1]].append(u)

                u1 = p_u[1]
                udd = self.parent[u1]

                v_key = tuple(sorted([w, u1]))
                v_idx = self.parallel_virtual_next_idx[v_key]
                self.parallel_virtual_next_idx[v_key] += 1
                e_virtual = Edge(w, u1, v_idx, is_virtual=True)

                # replace tree path w-u1 with virtual edge (w → u1)
                # TODO: Note, this messes with parent map. Check whether this is important (it is).
                self.trace_original_parent_patch[u1] = udd  # track the info on the original parent of u1
                self.parent[u1] = w
                self.ear[u1] = self.ear[u]
                self.virtual_edges.append(e_virtual)  # main graph virtual tree edge
                # remove u from p_u
                self.p[u] = [x for x in p_u if x != u]

            # if u = udd, CSδ(u) is already constructed by Lemma3.8(ii)(a)
            if u != udd:
                # Construct tree path u - udd
                tree_path = [udd]
                while True:
                    p = self.parent[tree_path[-1]]
                    tree_path.append(p)
                    if p == u:
                        break

                if self.t(self.p_anchor[u]) != u:
                    # Finalize CSδ(u) ; Lemma3.8(ii)(b) first case
                    self.cs[u].prepend(ConSeq([Path(tree_path), Path([u, udd])]))
                else:
                    # Finalize CSδ(u); Lemma3.8(ii)(b) second case
                    cs = self.cs[u]
                    self.cs[u] = ConSeq([Path(tree_path), Path([u, udd]), *cs.paths[cs.ih:], *cs.paths[:cs.ih]])

                # Path([u, udd]) is virtual.
                v_key = tuple(sorted([u, udd]))
                v_idx = self.parallel_virtual_next_idx[v_key]
                self.parallel_virtual_next_idx[v_key] += 1

                e_virtual = Edge(u, udd, v_idx, is_virtual=True)
                self.virtual_edges.append(e_virtual)  # finalized component virtual edge

        self.components.append(u)
        self.trace_mader[u] = self.materialize_component_cs_paths(u, extract_core=True)
        self.trace_absorbed.add(u)

    def absorb_ear(self, w, p_hat: Ear, p: List[int] | None, bchain_t: List):
        # absorb the entire P which is Pw or w + Pu with ˆP being ˆP(w) or ˆP(u), respectively
        # create construction sequence for ˆP ∪ Union δ(x_i) with i in [1, k]
        # σ(w) absorbs all σ(x) on the path
        if p is not None:
            content = p[1:]
            for x in content:
                self.trace_absorbed.add(x)
            if bchain_t:
                # P=u; u.bchain exists and is kept in bchain_t;
                # convert bchain_t to a cactus cycle and attach it to w
                self.cycle[w].append([*bchain_t])  # Mod: implicit attachment vertex
                bchain_t.clear()  # keep bchain_t reference intact
            else:  # not necessary because content is [] when bchain_t is present
                for x in content:
                    self.cycle[w].extend(self.cycle[x])
                    if self.tchain[x]:
                        self.cycle[w].append([*self.tchain[x]])  # Mod: implicit attachment vertex
                        self.tchain[x].clear()  # Not necessary, but we clean up absorbed chains.
                if self.bchain[p[-1]]:
                    self.cycle[w].append([*self.bchain[p[-1]]])  # Mod: implicit attachment vertex
                    # NOTE: New wrt to the paper. Unlike tchain clearing, this is mandatory to avoid duplicated cycles.
                    self.bchain[p[-1]].clear()

            self.sigma[w] = list(
                itertools.chain(self.sigma[w], itertools.chain.from_iterable(self.sigma[x] for x in content)))
            # CS(x) are concatenated in reverse order
            csp = itertools.chain.from_iterable(self.cs[x].paths for x in reversed(content))
        else:
            csp = []

        # ^P is the anchor
        if p_hat is not None:
            csp = itertools.chain([p_hat], csp)
        if self.lexlt(p_hat, self.p_anchor[w]):
            # ^P is the new anchor(w), so CS leads the construction sequence
            self.cs[w] = ConSeq(list(itertools.chain(csp, self.cs[w].paths)))
            self.p_anchor[w] = p_hat
        else:
            # P is not anchor(w), append CS to CSδ(w)
            self.cs[w] = ConSeq(list(itertools.chain(self.cs[w].paths, csp)))

    def absorb_path(self, w, p_w, inc_w: Set[Edge]):
        # absorb a section of the w-path Pw
        h = 0
        for e_inc in inc_w:
            # determine the lowest ancestor wh of x=e_inc.start on p_w
            # Note that incoming edges are directed to w, not from it.
            for wh in p_w[h + 1:]:
                if not self.is_ancestor(wh, e_inc.start):
                    break
                h += 1

        for j in range(1, h + 1):
            # Transfer all cactus cycles attached to w_j to w
            self.cycle[w].extend(self.cycle[p_w[j]])
            if self.tchain[p_w[j]]:
                # Convert w_j.tchain to a cactus cycle
                self.cycle[w].append([*self.tchain[p_w[j]]])  # Mod: implicit attachment vertex
                self.tchain[p_w[j]].clear()  # Not necessary, but we clean up absorbed chains.

        # Determine the absorbed vertex wl with the smallest anchor.
        l, p_anchor_l = find_min([self.p_anchor[p_w[j]] for j in range(0, h + 1)], self.lexlt)
        self.p_anchor[w] = p_anchor_l  # NOTE: New wrt to the paper.
        # Create CSδ(w) based on Lemma3.7
        cs_ih = 0
        if l != h:
            csp = self.cs[p_w[l]].paths + self.cs[p_w[h]].paths
            # Record the start of h-section of the CS to use later in Lemma3.8(ii)(b) second case.
            cs_ih = len(self.cs[p_w[l]].paths)
        else:
            csp = self.cs[p_w[h]].paths
        csp = list(
            itertools.chain(csp, itertools.chain.from_iterable(self.cs[p_w[j]].paths for j in range(0, h) if j != l)))
        self.cs[w] = ConSeq(csp, ih=cs_ih)

        # w absorbs w_j, 1 ≤ j ≤ h
        ws_to_absorb = [p_w[j] for j in range(1, h + 1)]
        for wj in ws_to_absorb:
            self.trace_absorbed.add(wj)
        self.sigma[w] = list(
            itertools.chain(self.sigma[w], itertools.chain.from_iterable(self.sigma[wj] for wj in ws_to_absorb)))
        self.p[w] = [wj for wj in self.p[w] if wj not in ws_to_absorb]

        # NOTE: New wrt to the paper.
        # Transfer bchain of an absorbed vertex if it exists.
        if ws_to_absorb:
            last_bchain = self.bchain[ws_to_absorb[-1]]
            # sanity check:
            if self.bchain[w]:
                raise ValueError("Non-empty self.bchain[w] when absorbing last_bchain")
            if any(self.bchain[x] for x in ws_to_absorb[:-1]):
                raise ValueError("Non-empty self.bchain[x] for internal x in absorb_path")
            if last_bchain and h != len(p_w) - 1:
                raise ValueError("bchain exists on not the terminal vertex in the path in absorb_path")
            if last_bchain:
                self.bchain[w] = [*last_bchain]
                last_bchain.clear()

    def s(self, ear: Ear):
        return ear.backedge.end

    def t(self, ear: Ear):
        s, e = ear.backedge.start, ear.backedge.end
        curr = ear
        while curr == ear:
            curr = self.ear[s]
            # None ear could correspond to a removed self-loop ear when a super-vertex is ejected.
            # if curr is None:
            #     raise ValueError("t(ear): encountered an edge with None ear")
            s, e = self.parent[s], s
        return e

    def is_ancestor(self, ancestor, v):
        # O(1) check using DFS interval containment:
        # ancestor is an ancestor of v iff dfs[ancestor] <= dfs[v] < dfs[ancestor] + desc_count[ancestor]
        return self.dfs[ancestor] <= self.dfs[v] < self.dfs[ancestor] + self.desc_count[ancestor]

    def lexlt(self, a: Edge | Ear, b: Edge | Ear):
        if b is None:
            return True
        if a is None:
            # raise ValueError("First edge is None")
            return False
        if isinstance(a, Ear):
            a = a.backedge
        if isinstance(b, Ear):
            b = b.backedge
        q, p = a.end, a.start
        y, x = b.end, b.start
        if self.dfs[q] < self.dfs[y]:
            return True
        if q == y:
            if not self.is_ancestor(p, x):
                if self.dfs[p] < self.dfs[x]:
                    return True
            if self.is_ancestor(x, p) and x != p:
                # p is proper descendant of x
                return True
        return False

    def materialize_decomposition(self):
        from utils import Decomposition

        return Decomposition(
            components=self.components,
            sigma={c: self.sigma[c] for c in self.components},
            cs=self.materialize_cs_paths(extract_core=False),  # for completeness
            bridges=[[b.start, b.end] for b in self.bridges],
            cycles={c: self.cycle[c] for c in self.components if self.cycle[c]},
            mader=self.materialize_cs_paths(extract_core=True)
        )

    def materialize_cs_paths(self, extract_core=True) -> Dict[int, List[List[int]]]:
        return {c: self.materialize_component_cs_paths(c, extract_core) for c in self.components if self.cs[c].paths}

    def materialize_component_cs_paths(self, component: int, extract_core=True) -> List[List[int]]:
        """Convert CS paths for a component to a list of vertex lists.

        Ensures that the construction sequence starts with 3 explicit paths forming
        the K32 core. If the first sequences include a cycle, splits it into branches
        at the common endpoints shared with adjacent paths.
        """
        paths_list = []
        seen_ears = set()

        # Core extraction state
        core_buffer = []
        pending_cycle = None
        pending_path = None
        core_complete = False

        for path in self.cs[component].paths:
            # Materialize the sequence
            if isinstance(path, Path):
                seq = path.vertices
            elif isinstance(path, Ear):
                ear = path  # alias for clarity

                path_vertices = [ear.backedge.end]
                v = ear.backedge.start
                curr_ear = ear
                while curr_ear == ear:
                    path_vertices.append(v)
                    p = self.parent[v]
                    if p is None:
                        # We reached the root vertex.
                        break
                    curr_ear = self.ear[v]
                    if curr_ear is None:
                        # Note, that None ear corresponds to a bridge or a removed self-loop ear.
                        # if p not in self.bridges:
                        #     raise ValueError("encountered an edge with None ear, and it is not a bridge")
                        break
                    if curr_ear in seen_ears:
                        # This corresponds to a parallel back-edge that is not a part of K32 core.
                        # We don't mark parallel edges identity specifically.
                        # So in this case we should not trace the ear back, because its tree path is already added by the original ear.
                        break
                    v = p

                seen_ears.add(ear)
                seq = path_vertices
            else:
                raise ValueError("Unknown path type")

            if core_complete or not extract_core:
                paths_list.append(seq)
                continue

            # Process sequence for K32 core extraction
            is_cycle = (seq[0] == seq[-1])
            if not is_cycle:
                # If we have a pending cycle, split it now
                if pending_cycle is not None:
                    cycle_paths = self._split_cycle_at_branches(pending_cycle, {seq[0], seq[-1]})
                    core_buffer.extend(cycle_paths)
                    pending_cycle = None
                    core_buffer.append(seq)
                elif pending_path is not None:
                    if pending_path[0] == seq[0]:
                        if pending_path[-1] == seq[-1]:
                            # Two paths form a cycle.
                            pending_cycle = [*pending_path, *reversed(seq[1:])]
                            pending_path = None
                        elif seq[-1] in set(pending_path):
                            # Branch points.
                            core_buffer.append(seq)
                            # pending_path is the same
                        elif pending_path[-1] in set(seq):
                            # Branch points.
                            core_buffer.append(pending_path)
                            pending_path = seq
                        else:
                            raise ValueError("unreachable")
                    elif pending_path[0] == seq[-1]:
                        if pending_path[-1] == seq[0]:
                            # Two paths form a cycle.
                            pending_cycle = [*pending_path, *seq[1:]]
                            pending_path = None
                        elif seq[0] in set(pending_path):
                            # Branch points.
                            core_buffer.append(seq)
                            # pending_path is the same
                        elif pending_path[0] in set(seq):
                            # Branch points.
                            core_buffer.append(pending_path)
                            pending_path = seq
                        else:
                            raise ValueError("unreachable")
                    else:
                        # Endpoints are completely different
                        if seq[0] in set(pending_path):
                            core_buffer.append(seq)
                            # pending_path is the same
                        elif pending_path[0] in set(seq):
                            core_buffer.append(pending_path)
                            pending_path = seq
                        else:
                            raise ValueError("unreachable")
                else:
                    pending_path = seq

            else:
                # It's a cycle
                if pending_cycle is not None:
                    raise ValueError("Encountered two cycles before K32 core is complete")
                elif pending_path is not None:
                    # We have branch points, can split immediately
                    cycle_paths = self._split_cycle_at_branches(seq, {pending_path[0], pending_path[-1]})
                    core_buffer.extend(cycle_paths)
                else:
                    # No branch points yet, store cycle for later
                    pending_cycle = seq

            # Check if core is complete
            if len(core_buffer) > 3:
                raise ValueError("K32 core has more than 3 paths")
            if len(core_buffer) == 3:
                # Align core paths and dump core.
                branch_point = core_buffer[0][0]
                for path in core_buffer:
                    if path[0] != branch_point:
                        path.reverse()
                paths_list.extend(core_buffer)
                core_complete = True

        return paths_list

    @staticmethod
    def _split_cycle_at_branches(cycle: List[int], branch_points: Set[int]) -> List[List[int]]:
        """Split a cycle into exactly 2 paths at branch points.

        Returns one direct path and one wrapped path between the branch points.
        """
        # Find all positions of branch points in the cycle
        branch_positions = [i for i, v in enumerate(cycle[:-1]) if v in branch_points]

        if len(branch_positions) != 2:
            raise ValueError(f"Cannot split cycle {cycle} with {len(branch_positions)} branch points: {branch_points}")

        # Take first two branch positions
        pos1 = branch_positions[0]
        pos2 = branch_positions[1]

        # Direct path from pos1 to pos2
        direct_path = cycle[pos1:pos2 + 1]

        # Wrapped path from pos2 back to pos1
        wrapped_path = cycle[pos2:] + cycle[1:pos1 + 1]

        # Return longest path first
        if len(direct_path) >= len(wrapped_path):
            return [direct_path, wrapped_path]
        else:
            return [wrapped_path, direct_path]


def find_min(items: Iterable, less: Callable):
    min_i = None
    min_item = None
    for i, item in enumerate(items):
        if min_item is None or less(item, min_item):
            min_i = i
            min_item = item
    if min_item is None:
        raise ValueError("Empty iterable")
    return min_i, min_item


# Tracing
def build_step(name, **data):
    return {
        "name": name,
        "data": data,
    }


if __name__ == '__main__':
    import json
    from t4racker import TTTTracker, TrackReplayer
    from utils import build_adjacency
    import examples

    example = examples.Tsin1_mod
    # example = examples.Tsin2
    # example = examples.Shuriken
    # example = examples.Ladder
    # example = examples.Tsin1
    # example = examples.Tsin2_mod
    # example = examples.SimpleCycle
    # example = examples.BchainAtRoot

    start_vertex = example.start_vertex

    # Setup
    graph = build_adjacency(example.graph.num_vertices, example.graph.edges)

    algo = Solver(graph, example.graph.num_vertices)

    tracker = TTTTracker()
    tracker.register_key_converters({
        Edge: lambda x: f"({x.start}, {x.end})"
    })
    tracked_fields = [
        'trace_current_path',
        'trace_major_step',
        'trace_absorbed',
        'trace_original_parent_patch',
        'trace_mader',

        'parent',
        'dfs',
        'p_hat',
        'p_anchor',
        'cs',
        'sigma',

        'p',

        'ear',

        'tchain',
        'bchain',
        'bchain_t',
        'cycle',

        'cnt',
        'bridges',
        'components',

        'virtual_edges',

        'inc',
    ]
    tracker.track(algo, tracked_fields=tracked_fields)
    tracker.capture_snapshot('initial_state')

    algo.certifying_3_edge_connectivity(start_vertex)

    tracker.capture_snapshot('final_state')

    decomposition = algo.materialize_decomposition()

    print("Construction sequences")
    for component, paths in decomposition.cs.items():
        print(f"{component}: {algo.sigma[component]}")
        for path in paths:
            print(path)
        print()

    print("Bridges")
    for bridge in decomposition.bridges:
        print(bridge)
    print()

    print(f"Number of Components: {len(decomposition.components)}")
    print(decomposition.components)
    print()

    print("Cycles")
    for v, cycles in decomposition.cycles.items():
        print(v, cycles)
    print()

    # print("Cycles")
    # i2l = {i: l for i, l in zip(range(1, example.graph.num_vertices + 1), "abcdfghijklmno")}
    # for v, cycles in decomposition.cycles.items():
    #     print(i2l.get(v, v), [[i2l.get(x, x) for x in cycle] for cycle in cycles])
    # print()

    print("Virtual Edges")
    for e in algo.virtual_edges:
        print(e)

    example_passed = decomposition.is_equal_strict(example.decomposition)
    print(f"Decomposition is correct: {example_passed}")

    print("\n" + "=" * 60)

    print(f"Total tracker steps: {len(tracker.steps)}")

    output_dir = "."
    # output_dir = "scratch_output"
    algorithm_name = "tsin-certifying-3-edge-connectivity"

    track_data = tracker.to_dict()
    with open(os.path.join(output_dir, f"{algorithm_name}_track_{example.name}.json"), 'w') as fp:
        json.dump(track_data, fp, indent=2)

    replayer = TrackReplayer(track_data)
    final_step_index = len(replayer.steps) - 1
    final_state = replayer.state_at(final_step_index)

    # # Save final state to JSON
    # with open('ecc_cert_final_state.json', 'w') as fp:
    #     json.dump(tracker.to_json(final_state), fp, indent=2)

    # Compare actual vs reconstructed state (forward)
    # Convert both to JSON format for consistent comparison
    actual_state_json = {field: tracker.to_json(getattr(algo, field)) for field in tracked_fields}
    final_state_json = {field: tracker.to_json(final_state[field]) for field in tracked_fields}

    forward_match = True
    for field in tracked_fields:
        if actual_state_json[field] != final_state_json[field]:
            print(f"✗ Forward mismatch in {field}")
            forward_match = False

    print(f"Forward reconstruction: {'✓ OK' if forward_match else '✗ FAILED'}")

    # Backward check: start from final state, apply steps backward to initial snapshot
    initial_snapshot_idx = replayer.snapshots[0]['step_idx']
    backward_state = json.loads(json.dumps(tracker.to_json(final_state)))
    backward_state = replayer._from_json(backward_state)
    # Only apply steps after the initial snapshot backward
    for step in reversed(replayer.steps[initial_snapshot_idx:]):
        replayer._apply_step(backward_state, step, reverse=True)

    # Compare with initial snapshot (convert from JSON)
    initial_state = json.loads(json.dumps(replayer.snapshots[0]['state']))
    initial_state = replayer._from_json(initial_state)
    backward_match = True
    for field in tracked_fields:
        if backward_state[field] != initial_state[field]:
            print(f"✗ Backward mismatch in {field}")
            backward_match = False

    print(f"Backward reconstruction: {'✓ OK' if backward_match else '✗ FAILED'}")
