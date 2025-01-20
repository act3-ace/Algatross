.. MO-MARL documentation master file, created by
   sphinx-quickstart on Mon Dec  2 23:03:56 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:html_theme.sidebar_secondary.remove: true
:html_theme.sidebar_primary.remove: true

.. template taken from Pandas & SciPy

`MO-MARL API Documentation </ascension/mo-marl>`_
=================================================

**Version**: |version|

The Ascension Team core library for Multi-Objective Multi-Agent Reinforcement Learning (MO-MARL) in PyTorch.

Features:

* Total algorithm customization from .yaml files.
* Total integration with `Ray <https://www.ray.io/>`_ library for highly scalable training and inference.
* Fast and stable distributed implementation of Multi-objective Asymmetric Island Model (MO-AIM) proposed by `Dixit and Tumer (2023) <https://doi.org/10.1145/3583131.3590524>`_.
* Integration with Heterogeneous-Agent Reinforcement Learning (HARL) proposed by `Zhong et al. (2024) <http://arxiv.org/abs/2304.09870>`_.
* Baseline implementations for PPO in RLlib and CleanRL.
* Distributed plotting and visualization for supported environments.
* Native filesystem-based multi-agent checkpoints. Navigate archipelago checkpoints entirely in a file browser.

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :img-top: :fontawesome:fa-solid\ fa-book-open
        :text-align: center
        :class-card: sd-card sd-main-index-card
        :shadow: md

        **User guide**
        ^^^

        The user guide provides in-depth information on the
        key concepts of MO-MARL with useful background information and explanation.

        +++

        .. button-link:: /ascension/mo-marl/user-guide
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card::
        :img-top: :fontawesome:fa-solid\ fa-code
        :text-align: center
        :class-card: sd-card sd-main-index-card
        :shadow: md

        **API reference**
        ^^^

        The reference guide contains a detailed description of
        the MO-MARL API. The reference describes how the methods work and which parameters can
        be used. It assumes that you have an understanding of the key concepts.

        +++

        .. button-ref:: mo-marl-api
            :color: secondary
            :click-parent:

            To the reference guide

    .. grid-item-card::
        :img-top: :fontawesome:fa-solid\ fa-person-running
        :text-align: center
        :class-card: sd-card sd-main-index-card
        :shadow: md

        **Getting started**
        ^^^

        Unsure where to begin with reinforcement learning or
        evolutionary algorithms? Check here for helpful resources
        and information.

        +++

        .. button-link:: /ascension/mo-marl/getting_started
            :color: secondary
            :click-parent:

            To the build guide

    .. grid-item-card::
        :img-top: :fontawesome:fa-solid\ fa-terminal
        :text-align: center
        :class-card: sd-card sd-main-index-card
        :shadow: md

        **Developer guide**
        ^^^

        Saw a typo in the documentation? Want to improve
        existing functionalities? The contributing guidelines will guide
        you through the process of improving MO-MARL.

        +++

        .. button-link:: /ascension/mo-marl/developer_guide
            :color: secondary
            :click-parent:

            To the development guide

.. toctree::
    :hidden:

    docs/index
    generated/index

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
