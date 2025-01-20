"""OnPolicy HARL runners."""

from algatross.environments.harl.runners.on_policy import OnPolicyBaseRunner


class OnPolicyHARunner(OnPolicyBaseRunner):
    """Runner for on-policy HA algorithms."""

    def train(self) -> dict:
        """Train the model.

        Returns
        -------
        dict
            The training results for the model.
        """
        train_infos = {}
        for agent in self.agents.values():
            train_infos.update(agent.train())  # type: ignore[attr-defined]
        return train_infos
