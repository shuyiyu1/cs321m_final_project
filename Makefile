.PHONY: reproduce figures tables test clean

reproduce:
	python scripts/run_all.py --output-dir outputs --seed 231

figures:
	python scripts/make_figures.py --output-dir outputs/figures --seed 231

tables:
	python scripts/make_tables.py --output-dir outputs/tables --seed 231

test:
	pytest -q

clean:
	rm -rf outputs/figures/*.pdf outputs/figures/*.png outputs/tables/*.csv outputs/tables/*.tex outputs/summary.json
