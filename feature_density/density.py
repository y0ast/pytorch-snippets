import torch

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov(x):
    n = x.shape[0]
    # Pre-multiply here to try to avoid overflow.
    x = (n - 1) ** (-0.5) * x

    # NOTE: on CPU with float16, mm is only single core
    res = x.t().mm(x)
    return res


# Default paramters compute GDA
# Set shared_cov to True for LDA
# Set shared_mean/shared_cov true for "marginal" feature density, to be able to compute a relative score
# Use diagonal_cov = True for Naive Bayes type classifier
def fit_gmm(
    embeddings, # Shape Dataset size by Features
    labels=None, # Shape Dataset size by Classes
    shared_mean=False,
    shared_cov=False,
    diagonal_cov=False,
):
    if not (shared_mean and shared_cov):
        assert (
            labels is not None
        ), "labels are required in all cases except double shared."

    embeddings = embeddings.cuda()
    labels = labels.cuda()

    with torch.no_grad():
        classes = torch.unique(labels, sorted=True)

        if shared_mean:
            mean_features = torch.mean(embeddings, dim=0)
        else:
            mean_features = torch.stack(
                [torch.mean(embeddings[labels == c], dim=0) for c in classes]
            )

        if shared_cov:
            if shared_mean:
                if diagonal_cov:
                    cov_features = torch.var(embeddings, dim=0, unbiased=True)
                else:
                    cov_features = centered_cov(embeddings - mean_features)
            else:
                if diagonal_cov:
                    cov_features = (
                        torch.stack(
                            [
                                torch.var(embeddings[labels == c], dim=0, unbiased=True)
                                for c in classes
                            ]
                        )
                        .mean(0, keepdim=True)
                        .expand(len(classes), -1, -1)
                    )
                else:
                    cov_features = (
                        torch.stack(
                            [
                                centered_cov(embeddings[labels == c] - mean_features[i])
                                for i, c in enumerate(classes)
                            ]
                        )
                        .mean(0, keepdim=True)
                        .expand(len(classes), -1, -1)
                    )
        else:
            assert (
                not shared_mean
            ), "shared mean + independent covariance is not supported"
            if diagonal_cov:
                cov_features = torch.stack(
                    [
                        torch.var(embeddings[labels == c], dim=0, unbiased=True)
                        for c in classes
                    ]
                )
            else:
                cov_features = torch.stack(
                    [
                        centered_cov(embeddings[labels == c] - mean_features[i])
                        for i, c in enumerate(classes)
                    ]
                )

        mean_features = mean_features.cpu().float()
        cov_features = cov_features.cpu().float()

        assert torch.all(
            torch.isfinite(cov_features)
        ), "Covariance contains inf or nan."
        if diagonal_cov:
            # This could be a MVN combined with:
            # https://pytorch.org/docs/stable/distributions.html#independent
            gmm = [
                torch.distributions.Normal(mean_features[i], cov_features[i].sqrt())
                for i in range(len(classes))
            ]
            return gmm, 0

        fit = False
        for jitter_eps in JITTERS:
            try:
                jitter = (
                    jitter_eps
                    * torch.eye(
                        cov_features.shape[1],
                        device=cov_features.device,
                    ).unsqueeze(0)
                )
                gmm = torch.distributions.MultivariateNormal(
                    loc=mean_features,
                    covariance_matrix=(cov_features + jitter),
                )
            except (RuntimeError, ValueError) as e:
                # Numerical issues -> increase jitter
                last_error = str(e)
                continue

            fit = True
            break

        assert fit, f"Creating gmm failed with jitter={jitter_eps} and: {last_error}"

    return gmm, jitter_eps
