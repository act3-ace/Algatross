---
title: Getting Started
---

To begin working with this library, you should make sure you have a firm grasp on some key concepts:

- [Reinforcement Learning](#reinforcement-learning)
- [Evolutionary Algorithms](#evolutionary-algorithms)
- [Tensor Frameworks](#tensor-frameworks)
- [Parallelism Frameworks](#parallelism-frameworks)

## Reinforcement Learning

The core approach of this library is built on reinforcement learning (RL), and evolutionary algorithms. As such , here are some resources to help familiarize you with reinforcement learning.

- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), Sutton & Barto
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/), OpenAI

## Evolutionary Algorithms

In addition to RL, evolutionary algorithms are deployed to build and evolve teams of agents. Notable resources include:

- [Non-Dominating Sort Genetic Algorithm-II](https://ieeexplore.ieee.org/document/996017) (NSGA-II), Deb

## Tensor Frameworks

Machine learning depends heavily on the linear-algebra concept of Tensors. In Python, there are a few libraries for working with tensors which efficiently handle tensor operations and gradient calculation. Every user should be familiar with these:

- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)

## Parallelism Frameworks

All of this heavy training can take a long time, especially when working around Pythons Global Interpreter Lock (GIL). This library relies on parallelization to speed up training. Users should familiarize themselves with the following libraries:

- [Ray](https://www.ray.io/)
- [Dask](https://www.dask.org/)
