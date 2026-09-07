# 單一原始碼,雙版本輸出。
#   make full    含 L3 實作補充  -> build/main-full.pdf
#   make paper   乾淨學術版(剝 L3) -> build/main-paper.pdf
#   make         = make full
#   make clean

LATEXMK = latexmk

.PHONY: all full paper status clean
.NOTPARALLEL:
all: full

full:
	$(LATEXMK) -xelatex -jobname=main-full \
	  -pretex='\def\IMPL{1}' -usepretex main.tex

paper:
	$(LATEXMK) -xelatex -jobname=main-paper \
	  -pretex='\def\IMPL{0}' -usepretex main.tex

status:
	$(LATEXMK) -xelatex -jobname=saccade-project-status-20260907 status-report.tex

clean:
	$(LATEXMK) -C
	$(LATEXMK) -C -jobname=saccade-project-status-20260907 status-report.tex
	rm -rf build
