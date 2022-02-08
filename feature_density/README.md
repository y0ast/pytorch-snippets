# Feature Density

There are several feature density methods used in the literature for uncertainty.
This snippet allows computing them easily.

It has special provisions for being very numerically stable and will try to find the minimum jitter value necessary for the inversion. The output of this function can be used with

This snippet computes all flavors of feature density:
- LDA: as suggested in "[A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/abs/1807.03888)" - also known as Mahalonobis distance
- GDA: as suggested in "[Deep Deterministic Uncertainty: A Simple Baseline](https://arxiv.org/abs/2102.11582)" - also known as DDU (without spectral norm)
- Marginal: as suggested in "[A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection](https://arxiv.org/abs/2106.09022)" - also known as relative Mahalonobis distance if combined with LDA.

Credit to [Andreas](https://github.com/BlackHC) and [Jishnu](https://github.com/omegafragger) for parts of this snippet!
