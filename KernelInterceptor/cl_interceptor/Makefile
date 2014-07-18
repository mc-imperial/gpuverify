CXX ?= g++
CXXFLAGS ?= -g -O0

# OPENCL INCLUDES BASED ON OS
UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
OPENCL_INC =
endif
ifeq ($(UNAME), Linux)
OPENCL_INC = -I$(CLDIR)/include
endif

default: libcl_interceptor.a

libcl_interceptor.a: cl_interceptor.o
	ar rcs $@ $<

cl_interceptor.o: cl_interceptor.cpp
	$(CXX) $(CXXFLAGS) $(OPENCL_INC) -c -o $@ $<

clean:
	rm -f cl_interceptor.o libcl_interceptor.a
