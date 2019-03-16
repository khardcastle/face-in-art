FILES = test.pdf

EXT = tex

SRCS = $(FILES:.pdf=.$(EXT))

LATEXMK_CMD = latexmk -bibtex

all: $(FILES)

%.pdf: %.$(EXT) FORCE
	$(LATEXMK_CMD) $<

clean:
	$(LATEXMK_CMD) -C $(SRCS)

cleanpart:
	$(LATEXMK_CMD) -c $(SRCS)

FILE ?= $(word 1,$(SRCS)) # If FILE is not set, pick a default
watch:
	$(LATEXMK_CMD) -pvc -interaction=nonstopmode $(FILE)

FORCE:

.PHONY: all clean cleanpart watch
