# CuratorNet: Visually-aware Recommendation of Art Images

### Manuel Cartagena<sup>1</sup>, Patricio Cerda<sup>1</sup>, Pablo Messina<sup>1</sup>, Felipe Del Río<sup>1</sup>, Denis Parra<sup>1</sup>

#### <sup>1</sup> Pontificia Universidad Católica de Chile

___
## Introduction

We introduce CuratorNet, a neural network architecture for visually-aware recommendation of art images.

CuratorNet is designed with the goal of maximizing generalization: the network has a fixed set of parameters that only need to be trained once, and thereafter the model is able to generalize to new users or items never seen before, without further training. This is achieved by leveraging visualcontent: items are mapped to item vectors through visual embeddings, and users are mapped to user vectors by aggregating the visualcontent of items they have consumed.

In this repository, we provide a TensorFlow implementation of CuratorNet.

The full paper is available at [arxiv link](), and is part of the ComplexRec workshop proceedings at the ACM RecSys 2020 conference.

## Folder structure

In `/experiments`, you can find all notebooks necessary for replicating our results.

In `/src`, you can find the CuratorNet implementation.

## License

This repository is MIT licensed (see [LICENSE](LICENSE)), except where noted otherwise. CHECK THIS IS THE ACTUAL CASE.
