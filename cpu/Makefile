# 1. Use Homebrew’s OpenMP-enabled clang++
CXX = /opt/homebrew/opt/llvm/bin/clang++

# 2. Include paths: your utility code + Boost headers
INCLUDE = -I../util -I../ -I/opt/homebrew/opt/boost/include

# 3. Compiler flags: optimization, C++11, OpenMP, validation macro
CFLAGS = -O3 -std=c++11 -fopenmp -DVALIDATE

# 4. Library paths and libraries: math, Boost.Thread/Boost.System, OpenMP, libomp
LIBDIRS = -L/opt/homebrew/opt/boost/lib
LIBS    = -lm -lboost_thread -lboost_system -fopenmp -lomp

# 5. Targets
all: pagerank
build: pagerank

# 6. Automatically find all .cpp files in current and parent dir
OBJ = $(patsubst %.cpp,%.o,$(wildcard *.cpp) $(patsubst %.cpp,%.o,$(wildcard ../*.cpp)))

# 7. Compile each .cpp to .o
%.o: %.cpp
	$(CXX) $(CFLAGS) $(INCLUDE) -c $< -o $@

# 8. Link everything into the pagerank executable
pagerank: $(OBJ)
	$(CXX) $(OBJ) $(CFLAGS) $(INCLUDE) $(LIBDIRS) -o $@ $(LIBS)

# 9. Clean up build artifacts
clean:
	rm -rf pagerank *.o
