TARGET = slim4pred
LIBS = -lm
CC = g++
CFLAGS = -g -Wall -fPIC
BUILDDIR := obj

ROOTFLAGS   := $(shell root-config --cflags)
ROOTLDFLAGS := $(shell root-config --ldflags)
ROOTLIBS    := $(shell root-config --glibs)

.PHONY: default all clean

default: $(TARGET)
all: default

OBJECTS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.hpp)

$(BUILDDIR)/%.o: %.cpp $(HEADERS)
	@mkdir -p $(BUILDDIR)
	$(CC) $(ROOTFLAGS) $(CFLAGS) $(ROOTLIBS) -I../CLI11/include -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(ROOTFLAGS) $(OBJECTS) -Wall $(LIBS) $(ROOTLIBS) -o $@

clean:
	@echo " Cleaning...";
	-rm -f *.o
	-rm -f $(TARGET)
