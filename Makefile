CONTAINER ?= icp
RUN = docker exec -w /icp-milestone-2 -e PYTHONPATH=. $(CONTAINER)

.PHONY: help profile tables sweep evaluate t6 t7

help:
	@echo "make profile   run the T6 sweep on the cluster, then rebuild every table and plot"
	@echo "make tables    rebuild every table and plot from the saved results (no sweep)"
	@echo "make sweep     T6 profiling sweep only (scripts/sweep.py, resumes finished points)"
	@echo "make evaluate  rerun milestone_2.ipynb (T0-T2 tables, T7 accuracy) from the OCR cache"
	@echo "make t6        T6 table, latency breakdown, accuracy vs p95 and vs throughput plots"
	@echo "make t7        T7 end-to-end table, Pareto plot, gold-input ablation"

profile: sweep tables

tables: evaluate t6 t7

sweep:
	python3 scripts/sweep.py

evaluate:
	$(RUN) jupyter nbconvert --to notebook --execute milestone_2.ipynb --output-dir results --output milestone_2_executed.ipynb --ExecutePreprocessor.timeout=-1

t6:
	$(RUN) python3 analysis/t6_tables.py

t7: t6
	$(RUN) python3 analysis/t7_end_to_end.py
	$(RUN) python3 analysis/t7_gold_ablation.py
