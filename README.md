# Softlearning
This SAC code is modified upon https://github.com/rail-berkeley/softlearning, where we drop the ray-based training style to a easy-reading run on a single process. Expert performances are run using this code.

SAC, 50 expert traj, Deterministic Policy in testing

| Envs | Mean | Std
| ----  | ----  | ----  |
| Pendulum | -147.5398 | 81.7622 |
| InvertedPendulum | 1000.0000 | 0.0000 |
| InvertedDoublePendulum | 9358.7842 | 0.3963 |
| Ant | 5404.5532 | 1520.4961 |
| Hopper | 3402.9494 | 446.4877 |
| Humanoid | 6043.9907 | 726.1788 |
| HalfCheetah | 13711.6445 | 111.4709 |
| Walker2d | 5639.3267 | 29.9715 |
| Swimmer | 139.2806 | 1.1204 |
| AntSlim | 5418.8721 | 946.7947 |
| HumanoidSlim | 5346.6181 | 712.2214 |
| SwimmerSlim | 339.2811 | 0.7625 |

P.S.: *Slim envs are those envs that use a wrapper who remove some dimension of the observation.
