"""Sequential Extreme Learning Machine modules."""

import torch

from torch import nn

from algatross.extreme_learning_machines.utils.ops import elm_forward_hidden_out_hook, torch_calc_regularized_elm_weights


class ELMSimpleSequential(nn.Sequential):
    """
    ELMSimpleSequential Sequential container for ELMs.

    Calls forward on each module in the container. Registers a forward hook for the last hidden layer. Provides functionality
    for training ELMS.

    Parameters
    ----------
    `*args`
        Positional arguments.
    `**kwargs`
        The config dict to pass to pass to the weight update step of this ELM module.
    """

    final_layer_hidden: str
    """The name of the final hidden layer of the network."""
    final_layer: str
    """The name of the output layer of the network."""

    def __init__(self, *args, **kwargs: dict):
        super().__init__(*args)
        last_hidden = None
        out_linear = None
        for mod_name, _ in reversed(list(self.named_children())):
            if out_linear is None:
                out_linear = mod_name
            else:
                last_hidden = mod_name
                break

        self.get_submodule(last_hidden).register_forward_hook(elm_forward_hidden_out_hook)  # type: ignore[arg-type]
        self.final_hidden_layer = last_hidden
        self.final_layer = out_linear
        self._a_inv: torch.Tensor = None
        self._previous_hidden_preds: torch.Tensor = None
        self.elm_learning_config = kwargs

    def learn_weights(self, targets: torch.Tensor, preds: torch.Tensor | None = None, weight_matrix: torch.Tensor | None = None) -> None:
        """
        Learn the output weights of the ELM using ``targets``.

        Parameters
        ----------
        targets : torch.Tensor
            The learning targets
        preds : torch.Tensor | None, optional
            The predictions to use for learning, :data:`python:None`. If None then the value stored in the last hidden layers
            ``_last_output`` buffer is used.
        weight_matrix : torch.Tensor | None, optional
            The weighting matrix to pass to the learning function, :data:`python:None`
        """
        preds = self.get_submodule(self.final_hidden_layer)._last_output if preds is None else preds  # noqa: SLF001
        new_weights, self._a_inv = torch_calc_regularized_elm_weights(
            hidden_outs=preds,
            targets=targets,
            current_elm_weights=self.get_submodule(self.final_layer).weight,
            a_inv=self._a_inv,
            weight_matrix=weight_matrix,
            previous_hidden_outs=self._previous_hidden_preds,
            **self.elm_learning_config,  # type: ignore[arg-type]
        )
        self.get_submodule(self.final_layer).set_parameters(new_weights.transpose(1, 0))
        self._previous_hidden_preds = preds
