CC=g++
DYNET_DIR = ./dynet
EIGEN = ./eigen
DYNET_BUILD_DIR=$(DYNET_DIR)/build
INCS=-I$(DYNET_DIR) -I$(DYNET_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(DYNET_BUILD_DIR)/dynet/ -L$(PREFIX)/lib
FINAL=-ldynet -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-lgdynet -lboost_regex -lboost_serialization -lboost_program_options -lcudart -lcublas -lpthread -lrt
CFLAGS=-std=c++11 -Ofast -g -pipe
#CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train $(BINDIR)/show_grammar $(BINDIR)/parse

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o io.o pcfglm.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/show_grammar: $(addprefix $(OBJDIR)/, show_grammar.o io.o pcfglm.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/parse: $(addprefix $(OBJDIR)/, parse.o cyk_parser.o io.o syntax_tree.o pcfglm.o mlp.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
