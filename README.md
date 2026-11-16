# ICP Milestone 2
Deploying, Monitoring, and Profiling Pipeline Variants

Setting up in Chameleon Environment
1. Log in to Chameleon's hosted JupyterHub.
2. Upload setup/setup_environment.ipynb from this repo into own Chameleon Jupyter environment (or clone this repo directly from a terminal inside that environment).
3. Open setup/setup_environment.ipynb and run the cells in order

Regenerating the tables and plots
Run from the repo root on the VM, with the dev container `icp` running (setup/start_dev_container.sh).
- `make tables` rebuilds every table and plot from the saved results: reruns milestone_2.ipynb from the OCR cache (T0-T2 tables, T7 accuracy), then analysis/t6_tables.py and analysis/t7_*.py. Outputs go to results/ and plots/.
- `make profile` runs the T6 sweep (scripts/sweep.py) on the cluster first, then everything in `make tables`.
- `make help` lists the single steps (sweep, evaluate, t6, t7).
