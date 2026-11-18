# Protocol audit changes

- Corrected native Crypto WRMSE total, last-feature evaluation, asset weights, test origin/stride and train-window split.
- Corrected native Player Trajectory windowing, per-trajectory normalization and recursive 30-step evaluation.
- Added ragged irregular-data export, official split seed and macro masked MSE/MAE.
- Corrected M4 frequency horizons, native window/normalization conventions and executable Quarterly validation fraction; fail on unresolved short histories.
- Corrected Shallow Water stride-one training windows and 11-block rollout.
- Separated native KRNO and local SHIFT/control identities; blocked unverified historical reproduction.
- Added dataset-specific reconstruction settings, uncertainty-head safeguards, common-likelihood control launcher, noise/dropout controls, donor-map evaluation, capacity-count matching, synthetic informative-region task and regression tests.
- No real dataset, checkpoint or published score is included. See docs/REPRODUCTION_STATUS.md for outstanding work.
