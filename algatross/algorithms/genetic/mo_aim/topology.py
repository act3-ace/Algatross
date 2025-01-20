"""A module defining classes for the topology of the archipelago borrowed from PaGMOs structure."""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

import torch

from torch import nn

from algatross.utils.types import IslandID, IslandSample, IslandSpec, IslandTypeStr, MainlandID


class MOAIMTopology(nn.Module):  # noqa: PLR0904
    """
    A module definining the connection weights between islands in an archipelago.

    Parameters
    ----------
    alpha : float, optional
        The learning rate for topology parameters, default is :python:`3e-4`
    nu : float, optional
        The entropy regularization control parameter, default is :python:`0.01`
    dtype : torch.dtype, optional
        The datatype for the parameters, default is :class:`torch.float64`
    `*args`
        Additional positional arguments.
    `**kwargs`
        Additional keyword arguments.
    """

    archipelago_topology: torch.Tensor
    """The topology for the entire archipelago."""
    nu: float
    """The entropy regularization parameter."""
    alpha: float
    """The learning rate for the softmax weights."""

    _island_indices: list[IslandID]
    _mainland_indices: list[MainlandID]

    def __init__(self, alpha: float = 3e-4, nu: float = 0.01, dtype: torch.dtype = torch.float64, *args, **kwargs):
        super().__init__()
        self.nu = nu
        self.alpha = alpha
        self._island_indices = []
        self._mainland_indices = []
        self.dtype = dtype
        self.register_buffer("archipelago_topology", torch.empty(0, 0, dtype=self.dtype))

    def __len__(self):  # noqa: D105
        return self.n_mainlands + self.n_islands

    # def __reduce__(self):
    #     deserializer = MOAIMTopology  # noqa: ERA001
    #     serialized_data = (self.archipelago_topology, self.nu, self.alpha)  # noqa: ERA001
    #     return deserializer, serialized_data  # noqa: ERA001

    def push_back_island(self, n: int = 1):
        """Add ``n`` islands to the topology.

        Parameters
        ----------
        n : int, optional
            The number of islands to add, by default 1
        """
        self._island_indices.extend(list(range(len(self), len(self) + n)))
        self._island_indices = sorted(self._island_indices)

        arch_cnxn = self.construct_topology(n, torch.zeros, torch.ones, self.__island_index_tensor[-n:], self.archipelago_topology)
        self.register_buffer("archipelago_topology", arch_cnxn)

    def push_back_mainland(self, n: int = 1):
        """Add ``n`` mainlands to the topology.

        Parameters
        ----------
        n : int, optional
            The number of mainlands to add, by default 1
        """
        self._mainland_indices.extend(list(range(len(self), len(self) + n)))
        self._mainland_indices = sorted(self._mainland_indices)

        arch_cnxn = self.construct_topology(n, torch.ones, torch.zeros, self.__mainland_index_tensor[-n:])
        self.register_buffer("archipelago_topology", arch_cnxn)

    def swap_island_to_mainland(self, island_indices: MainlandID | list[MainlandID] = -1, in_place: bool = False):
        """Convert the topology of nodes set up as MO-AIM islands to be mainlands.

        Since Pagmo doesn't pass any arguments or keywords to :meth:`push_back` we can't initialize mainlands from
        the call to the archipelagos :meth:`push_back` method. Therefore we have to provide a way to convert a node
        which has been set up as an island (default) to behave as a mainland.

        The softmax distribution for each mainland has the corresponding island terms removed and is re-registered.

        New softmax distributions for each converted mainland is created and registered.

        The indices from the list of island indices are moved to the mainland index list.

        The connection weights are converted so that there is no migration between the new mainlands and other
        mainlands; the connection weight between the new mainlands and existing islands is initialized to a
        vector of ones.

        Parameters
        ----------
        island_indices : MainlandID | list[MainlandID], optional
            The node indices to be converted to mainlands, by default -1 (most recently added island). Node indices
            may be positive or negative with negative indices indicating the island index from the end of the list.
        in_place : bool, optional
            Whether or not the swap operation should take place in-place
        """
        if isinstance(island_indices, int):
            island_indices = [island_indices]

        # change negative values to a positive index
        for idx, isl in enumerate(island_indices):
            if isl < 0:
                island_indices[idx] = self.n_islands + isl

        if in_place:
            self.__in_place_island_mainland_swap(island_indices)
        else:
            mainland_indices = [isl for isl in self._island_indices if isl not in island_indices]
            self.set_archipelago(island_indices, mainland_indices)

    def __in_place_island_mainland_swap(self, island_indices: list[MainlandID]):
        # remove the islands' weights from each mainland softmax
        for m in range(self.n_mainlands):
            old_param = self.get_parameter(f"mainland_{m}_softmax")
            del self._parameters[f"mainland_{m}_softmax"]
            new_param = old_param.data[[isl for isl in range(self.n_islands) if isl not in island_indices]]
            self.register_parameter(f"mainland_{m}_softmax", nn.Parameter(new_param))

        # create paramters for the new mainlands
        repl_len = len(island_indices)
        for m in range(repl_len):
            new_param = torch.ones(self.n_islands - repl_len, dtype=self.dtype)
            self.register_parameter(f"mainland_{self.n_mainlands + m - repl_len}_softmax", nn.Parameter(new_param))

        # move the island indices to the mainland indices
        for isl in island_indices:
            self._island_indices.remove(isl)
            self._mainland_indices.append(isl)

        self._island_indices = sorted(self._island_indices)
        self._mainland_indices = sorted(self._mainland_indices)

        # create new connection weights
        ml_isl_cnxn = torch.ones(self.n_islands, repl_len, dtype=self.dtype)
        ml_ml_cnxn = torch.zeros(self.n_mainlands, repl_len, dtype=self.dtype)

        # replace the old weights
        arch_cnxn = self.update_topology(
            repl_len,
            island_edge_weights=ml_isl_cnxn,
            mainland_edge_weights=ml_ml_cnxn,
            index_tensor=torch.tensor(island_indices, dtype=torch.long),
            topology=self.get_buffer("archipelago_topology"),
        )
        self.register_buffer("archipelago_topology", arch_cnxn)

    def push_back(self, n: int = 1, island_class: IslandTypeStr = "island"):
        """Add ``n`` objects of the given ``island_class``.

        Parameters
        ----------
        n : int, optional
            The number of islands to push bacl, by default 1.
        island_class : IslandTypeStr, optional
            The class of the island being pushed back, by default "island"
        """
        if island_class == "island":
            self.push_back_island(n)
        elif island_class == "mainland":
            self.push_back_mainland(n)

    def get_connections(self, n: IslandID | MainlandID) -> tuple[np.ndarray, np.ndarray]:
        """Get the connections (edge weights) from all other islands in the topology coming into the island at index ``n``.

        Parameters
        ----------
        n : IslandID | MainlandID
            The index of the island whose connections we want

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            The connection weights to- and from- the island
        """
        ind = self.archipelago_topology[n].argwhere().numpy()
        weights = self.archipelago_topology[ind].numpy()
        return ind, weights

    def get_extra_info(self) -> dict[str, Any]:
        """Get extra info about the topology.

        Returns
        -------
        dict[str, Any]
            Extra info about the topology.
        """
        return {"islands": self.island_indices, "mainlands": self.mainland_indices}

    def get_softmax_fn(self) -> Callable[[Sequence[int]], dict[MainlandID, list[IslandSample]]]:
        """Construct a softmax function from the weight parameters of each mainland.

        Returns
        -------
        Callable[[Sequence[int]], dict[MainlandID, list[IslandSample]]]
            A function which takes a sequence of integers and returns a dictionary mapping mainland indices to a list of
            :class:`~algatross.utils.types.IslandSample`.
        """
        rows, cols = self.__get_slice_tensors(self.n_mainlands, self.n_islands, self.__mainland_index_tensor, self.__island_index_tensor)
        weights = self.get_buffer("archipelago_topology")[rows, cols]

        def softmax(
            num_samples: Sequence[int],
            generator: torch.Generator | None = None,
            weights: torch.Tensor = weights,
            ma_idx_list: list[MainlandID] = self._mainland_indices,
            isl_idx_list: list[IslandID] = self._island_indices,
        ) -> dict[MainlandID, list[IslandSample]]:
            samples = {}
            for ma, (w, n) in enumerate(zip(weights, num_samples, strict=True)):
                if n > 0:
                    m = torch.multinomial(torch.softmax(w, dim=0), n, replacement=True, generator=generator)
                    isl_idx, count = m.unique(return_counts=True)
                    samples[ma_idx_list[ma]] = [
                        IslandSample(island=isl_idx_list[isl], migrants=isl_cnt) for isl, isl_cnt in zip(isl_idx, count, strict=True)
                    ]
                else:
                    samples[ma_idx_list[ma]] = []
            return samples

        return softmax

    def reset_softmax_for(self, land_index: IslandID | MainlandID):
        """
        Reset the softmax for a land mass to a uniform distribution.

        Parameters
        ----------
        land_index : IslandID | MainlandID
            The index of the land (island/mainland) to reset.
        """
        if land_index in self._island_indices:
            index = torch.tensor([self.island_indices[self.island_indices.index(land_index)]], dtype=torch.long)

            new_isl_weights = torch.zeros((self.n_islands, 1), dtype=self.dtype)
            new_ml_weights = torch.ones((self.n_mainlands, 1), dtype=self.dtype)
        else:
            index = torch.tensor([self.mainland_indices[self.mainland_indices.index(land_index)]], dtype=torch.long)

            new_isl_weights = torch.ones((self.n_islands, 1), dtype=self.dtype)
            new_ml_weights = torch.zeros((self.n_mainlands, 1), dtype=self.dtype)

        rows_isl, cols_isl = self.__get_slice_tensors(self.n_islands, 1, self.__island_index_tensor, index)
        rows_ml, cols_ml = self.__get_slice_tensors(self.n_mainlands, 1, self.__mainland_index_tensor, index)

        self.get_buffer("archipelago_topology")[rows_isl, cols_isl] = new_isl_weights
        self.get_buffer("archipelago_topology")[rows_ml, cols_ml] = new_ml_weights

    def sanitize_topology(self):
        """Fully rebuilds the archipelago topology.

        All buffers and parameters are cleared
        """
        del_buffers = list(self._buffers.keys())
        del_params = list(self._parameters.keys())

        # clear all old buffers and parameters
        for k in del_buffers:
            del self._buffers[k]
            if hasattr(self, k):
                delattr(self, k)
        for k in del_params:
            del self._params[k]
            if hasattr(self, k):
                delattr(self, k)

        arch_cnxn = torch.empty((len(self), len(self)), dtype=self.dtype)
        arch_cnxn = self.construct_topology(self.n_islands, torch.zeros, torch.ones, self.__island_index_tensor, arch_cnxn)
        arch_cnxn = self.construct_topology(self.n_mainlands, torch.ones, torch.zeros, self.__mainland_index_tensor, arch_cnxn)

        self.register_buffer("archipelago_topology", arch_cnxn)

    def construct_topology(
        self,
        n: int,
        isl_cnxn_fn: Callable,
        ml_cnxn_fn: Callable,
        indices: torch.Tensor,
        topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Construct a topology which adds or modifies ``n`` connections.

        This will update the ``topology`` by modifying ``n`` connections at the given indices. The ``isl_cnxn_fn`` and ``ml_cnxn_fn`` will
        be used to generate new tensors of shape :python:`(n_islands, n)` and :python:`(n_mainlands_n)` respectively.

        Parameters
        ----------
        n : int
            The number of new or modified connections
        isl_cnxn_fn : Callable
            The function for initializing connections to islands
        ml_cnxn_fn : Callable
            The function for initializing connections to mainlands
        indices : torch.Tensor
            The indices which should be updated or added
        topology : torch.Tensor | None, optional
            The topology to be updated

        Returns
        -------
        torch.Tensor
            The connection matrix representing the edge weights
        """
        isl_cnxn = isl_cnxn_fn(self.n_islands, n, dtype=self.dtype)
        ml_cnxn = ml_cnxn_fn(self.n_mainlands, n, dtype=self.dtype)
        return self.update_topology(n, island_edge_weights=isl_cnxn, mainland_edge_weights=ml_cnxn, index_tensor=indices, topology=topology)

    def update_topology(
        self,
        n: int,
        island_edge_weights: torch.Tensor,
        mainland_edge_weights: torch.Tensor,
        index_tensor: torch.Tensor | None = None,
        topology: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Modify or add ``n`` vertices to the topology.

        Parameters
        ----------
        n : int
            The number of connections to add/update
        island_edge_weights : torch.Tensor
            The migration rate between the new vertices and the islands
        mainland_edge_weights : torch.Tensor
            The migration rate between the new vertices and the mainlands
        index_tensor : torch.Tensor | None, optional
            The indices to update, defaults to :data:`python:None`. If None, then the last ``n`` indices are used.
        topology : torch.Tensor | None, optional
            The topology to update, defaults to :data:`python:None`. If None, then the existing topology is used and it is assumed that
            connections are being added.

        Returns
        -------
        torch.Tensor
            The connection matrix representing the edge weights

        Raises
        ------
        ValueError
            if the shapes of the index tensor are incompatible
        """
        if index_tensor is not None and index_tensor.shape[0] != n:
            msg = f"Got incompatable shapes in dimension 0 for argument n_range={index_tensor.shape[0]} with n={n}"
            raise ValueError(msg)
        if index_tensor is None:
            index_tensor = torch.arange(-n, 0, 1, dtype=torch.long)

        # index tensors
        arch_idx_m, arch_idy_m = self.__get_slice_tensors(self.n_mainlands, n, self.__mainland_index_tensor, index_tensor)
        arch_idx_i, arch_idy_i = self.__get_slice_tensors(self.n_islands, n, self.__island_index_tensor, index_tensor)

        arch_cnxn = self.get_buffer("archipelago_topology") if topology is None else topology
        if topology is None:
            # no topology was given so we are adding n rows and columns to the archipelago topology
            # the n mainlands have already been appended to the mainland indices so we first use len(self) - n columns
            arch_cnxn = torch.cat([arch_cnxn, torch.empty(n, len(self) - n, dtype=self.dtype)], dim=0)
            arch_cnxn = torch.cat([arch_cnxn, torch.empty(len(self), n, dtype=self.dtype)], dim=1)

        # row = weight from other to vertex (incoming)
        # col = weight from vertex to other (outgoing)
        arch_cnxn[arch_idx_m, arch_idy_m] = mainland_edge_weights
        arch_cnxn[arch_idx_i, arch_idy_i] = island_edge_weights

        return arch_cnxn

    def set_archipelago(self, island_indices: Sequence[IslandID], mainland_indices: Sequence[MainlandID]):
        """Set the entire archipelago topology at once from the given sequence of island and mainland ids.

        Parameters
        ----------
        island_indices : Sequence[IslandID]
            Indices of the islands to set
        mainland_indices : Sequence[MainlandID]
            Indices of the mainlands to set

        Raises
        ------
        ValueError

            - If any of the ``island_indices`` are found in ``mainland_indices``
            - If any of the ``mainland_indices`` are found in ``island_indices``
        """
        for isl in island_indices:
            if isl in mainland_indices:
                msg = f"Island and mainland indicese may not overlap. Found island {isl} in mainland indices."
                raise ValueError(msg)
        for isl in mainland_indices:
            if isl in island_indices:
                msg = f"Island and mainland indicese may not overlap. Found mainland {isl} in island indices."
                raise ValueError(msg)
        self._island_indices = sorted(island_indices)
        self._mainland_indices = sorted(mainland_indices)
        self.sanitize_topology()

    @property
    def island_indices(self) -> list[IslandID]:
        """
        The vertices in the topology which correspond to the islands.

        Returns
        -------
        list[IslandID]
            The island indices in this topology
        """
        return self._island_indices

    @island_indices.setter
    def island_indices(self, x: Iterable[IslandID]):
        self._island_indices = sorted(x)
        self.sanitize_topology()

    @property
    def mainland_indices(self) -> list[MainlandID]:
        """
        The vertices in the topology which correspond to the mainlands.

        Returns
        -------
        list[MainlandID]
            The mainland indices in this topology
        """
        return self._mainland_indices

    @mainland_indices.setter
    def mainland_indices(self, x: Iterable[MainlandID]):
        self._mainland_indices = sorted(x)
        self.sanitize_topology()

    @property
    def n_islands(self) -> int:
        """
        Number of verticies which behave as islands in the topology.

        Returns
        -------
        int
            Number of verticies which behave as islands in the topology.
        """
        return len(self._island_indices)

    @property
    def n_mainlands(self) -> int:
        """
        Number of verticies which behave as mainlands in the topology.

        Returns
        -------
        int
            Number of verticies which behave as mainlands in the topology.
        """
        return len(self._mainland_indices)

    @property
    def archipelago(self) -> tuple[list[IslandID], list[MainlandID]]:
        """
        The topology for the entire archipelago.

        Returns
        -------
        tuple[list[IslandID], list[MainlandID]]
            The topology for the entire archipelago.
        """
        return self._island_indices, self._mainland_indices

    @archipelago.setter
    def archipelago(
        self,
        arch: (
            tuple[Iterable[IslandID], Iterable[MainlandID]]
            | Mapping[Literal["islands", "mainlands"], Iterable[IslandID] | Iterable[MainlandID]]
            | Iterable[IslandSpec]
            | Sequence[IslandTypeStr]
        ),
    ):
        islands: list[IslandID] = []
        mainlands: list[MainlandID] = []

        try:
            arch_dict: dict[str, Iterable[IslandID] | Iterable[MainlandID]] = dict(arch)  # type: ignore[arg-type]
            if len(arch_dict) != len(list(arch)):
                raise ValueError  # noqa: TRY301
            islands.extend(arch_dict["islands"])
            mainlands.extend(arch_dict["mainlands"])
        except ValueError as err:

            def append(string, idx, err=err):
                if string == "island":
                    islands.append(idx)
                elif string == "mainland":
                    mainlands.append(idx)
                else:
                    msg = f"Expected 'island' or 'mainland', got {string}."
                    raise ValueError(msg) from err

            arch_iter = iter(arch)
            isl = next(arch_iter)

            if isl in {"island", "mainland"}:
                idx = 1
                # we got a sequence of IslandTypeStr
                # the position of IslandTypeStr is the index of an island of the corresponding type
                append(isl, 0)
                for isl in arch_iter:
                    append(isl, idx)
                    idx += 1
            else:
                it = iter(isl)
                first = next(it)
                if isinstance(first, IslandID):
                    # we got an iterable (tuple) of Iterable[IslandID], Iterable[MainlandID]
                    # add the first item as well as all the remaining items in the iterator
                    islands.append(first)
                    islands.extend(it)  # type: ignore[arg-type]
                    # move to the iterator of the second item (mainland IDs)
                    mainlands.extend(next(arch_iter))  # type: ignore[arg-type]
                else:
                    # we got a iterable of IslandSpec
                    second = next(it)
                    if first in {"island", "mainland"} and isinstance(second, IslandID | MainlandID):
                        append(first, second)
                        for isl in arch_iter:
                            it = iter(isl)
                            first = next(it)
                            second = next(it)
                            append(isl, idx)
                    else:
                        msg = "Could not construct a topology from the given specs."
                        raise ValueError(msg) from err

        self.set_archipelago(islands, mainlands)

    @property
    def __island_index_tensor(self) -> torch.Tensor:
        return torch.tensor(self._island_indices, dtype=torch.long)

    @property
    def __mainland_index_tensor(self) -> torch.Tensor:
        return torch.tensor(self._mainland_indices, dtype=torch.long)

    @staticmethod
    def __get_slice_tensors(m: int, n: int, slice_tensor: torch.Tensor, index_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # island index tensors
        idx = torch.repeat_interleave(slice_tensor, n).view(m, n)
        idy = index_tensor.repeat(m).view(m, n)
        return idx, idy

    def optimize_softmax(self, utility: torch.Tensor, mainlands: list[MainlandID] | MainlandID | None = None) -> dict[str, Any]:
        """Calculate the entropy-regularized softmax loss for the topology and update the parameters.

        Parameters
        ----------
        utility : torch.Tensor
            The conspecific utility values
        mainlands : list[MainlandID] | MainlandID | None, optional
            The mainlands to update, :data:`python:None`

        Returns
        -------
        dict[str, Any]
            Update result info
        """
        # utility: [M x I] tensor of conspecific utilities
        results: dict[str, Any] = {}
        if mainlands is None:
            mainlands = list(self._mainland_indices)
        if not isinstance(mainlands, list):
            mainlands = [mainlands]
        rows, cols = self.__get_slice_tensors(
            len(mainlands),
            self.n_islands,
            torch.tensor([self.mainland_indices[self.mainland_indices.index(ml)] for ml in mainlands], dtype=torch.long),
            self.__island_index_tensor,
        )
        sm = self.get_buffer("archipelago_topology")[rows, cols]
        sm = nn.functional.softmax(sm, dim=1)
        with torch.no_grad():
            # create the broadcasted tensors X and X.T
            s_stack = sm.unsqueeze(-1).broadcast_to(*sm.shape, sm.shape[-1]).clone()
            s_tpose = s_stack.transpose(-1, -2).clone()
            s_stack = torch.eye(sm.shape[-1]) - s_stack

            # Jacobian
            jacobi = s_tpose * s_stack

            # Update the softmax weights
            d = jacobi.mul((utility - torch.log(sm).mul(self.nu)).unsqueeze(1)).sum(-1)
            self.get_buffer("archipelago_topology")[rows, cols] += self.alpha * d

        results["loss"] = {
            "total": d.sum().float(),
            **{f"mainland/{m}": dt.sum(dim=-1) for m, dt in zip(mainlands, d.detach().float(), strict=True)},
        }
        results["softmax_weights"] = {f"mainland/{m}": wt for m, wt in zip(mainlands, sm.detach().float(), strict=True)}
        return {"topology/optimize_softmax": results}

    def __repr__(self):  # noqa: D105
        return f"{self.__class__.__name__}(alpha={self.alpha}, nu={self.nu}, islands={self.n_islands}, mainlands={self.n_mainlands})"
