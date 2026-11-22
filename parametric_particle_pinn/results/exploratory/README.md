# CPU training experiments

These runs preceded the accepted [CPU run](../cpu). The configurations, logs,
and weights are retained as recorded. The earlier checkpoints used intermediate
architectures or training code and require their matching implementation to
load. Use the accepted run's checkpoints with the current API.

| Run | Approach | Independent trajectory checks |
| --- | --- | --- |
| v1 | Tanh time network with structured release features | Development, 4/32 accepted |
| v2 | Conditioned Chebyshev features with joint optimization | Development residuals only |
| v3 | Transverse warm-up followed by joint optimization | Development residuals only |
| v4 | Separate Adam stages followed by joint L-BFGS | Development, 23/32 accepted |

No final-test results were used to choose the accepted model.
