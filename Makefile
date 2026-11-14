# 二维射线层析重建：构建/测试/运行入口
# 环境：Python 3.11，依赖见 requirements.txt（numpy、scipy）
PYTHON ?= .venv/bin/python
OUT    ?= out

.PHONY: help venv install test build run-full run-limited run-tiny clean

help:
	@echo "make venv       创建虚拟环境并安装依赖"
	@echo "make test       运行全部单元测试（含 tiny 手算与独立积分交叉校验）"
	@echo "make build      测试 + 运行完整/有限角度/tiny 三组重建并落盘"
	@echo "make run-full   仅跑完整角度案例"
	@echo "make run-limited 仅跑有限角度案例"
	@echo "make run-tiny   仅跑 tiny 手算场景"
	@echo "make clean      删除 out/ 与缓存"

.venv/bin/python:
	python3 -m venv .venv
	$(PYTHON) -m pip install --quiet --upgrade pip
	$(PYTHON) -m pip install --quiet -r requirements.txt

venv install: .venv/bin/python

test: .venv/bin/python
	$(PYTHON) -m unittest discover -s tests -v

run-full: .venv/bin/python
	$(PYTHON) -m recon2d --rays fixtures/rays.csv --grid fixtures/grid.json \
		--out $(OUT)/full --name full

run-limited: .venv/bin/python
	$(PYTHON) -m recon2d --rays fixtures/limited-rays.csv --grid fixtures/grid.json \
		--out $(OUT)/limited --name limited

run-tiny: .venv/bin/python
	$(PYTHON) -m recon2d --rays fixtures/tiny/rays.csv --grid fixtures/tiny/grid.json \
		--out $(OUT)/tiny --name tiny

build: test run-full run-limited run-tiny
	@echo "全部产物见 $(OUT)/{full,limited,tiny}/"

clean:
	rm -rf $(OUT) **/__pycache__ recon2d/__pycache__ tests/__pycache__
