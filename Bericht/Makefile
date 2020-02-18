#
# build thesis for Automotive Power Measurement with Rotation, 
#GPS and On-Board-Diagnostics Data
#
#2019 Michael Schmid
# 
all:	Paper

TEXFILES = main.tex								\
	0-abstract/0-abstract.tex						\
	00-abbreviations/00-abbreviations.tex					\
	1-intro/1-intro.tex 							\
	2-power_meas/2-power_meas.tex	

Paper: main.pdf

main.ind:	$(TEXFILES) 
	touch main.ind
	pdflatex main.tex
	makeindex main.idx

main.pdf:	$(TEXFILES) main.ind 
	pdflatex main.tex

clean:
	rm -f main.aux main.ind main.log main.toc
