#include "Meta.h"
#include "GraphVec.h"
#include "Profiler.h"
#include "PPRCPUPowVec.h"
#include "SlidingGraphVec.h"
#include "PPRCPUMTCilkRev.h"
#include "PPRCPUMTCilkRevVariants.h"
#include "Arguments.h"
#include <ctime>
#include <cstdlib>
#include <omp.h>    // For OpenMP
#include <iostream> // For std::cout

int main(int argc, char *argv[])
{
    // Minimal OpenMP "hello world" test
    std::cout << "Testing OpenMP..." << std::endl;
#pragma omp parallel
    {
#pragma omp critical
        {
            std::cout << "Hello from thread " << omp_get_thread_num()
                      << " out of " << omp_get_num_threads() << " threads." << std::endl;
        }
    }
    std::cout << "OpenMP test finished." << std::endl;
    // End of OpenMP test

    ArgumentsParser(argc, argv);
    PrintArguments();

    GraphVec *graph = NULL;
    if (gIsDynamic)
    {
        graph = new SlidingGraphVec(gDataFileName, gIsDirected);
    }
    else
    {
        graph = new GraphVec(gDataFileName, gIsDirected);
    }

    Profiler::InitProfiler(1, PROFILE_PHASE_NUM, PROFILE_COUNT_TYPE_NUM);

    PPRCPUMTCilk *ppr = NULL;
    if (gAppType == REVERSE_PUSH)
    {
        if (gVariant == OPTIMIZED)
            ppr = new PPRCPUMTCilkRev(graph);
        else if (gVariant == FAST_FRONTIER)
            ppr = new PPRCPUMTCilkRevFF(graph);
        else if (gVariant == EAGER)
            ppr = new PPRCPUMTCilkRevEager(graph);
        else if (gVariant == VANILLA)
            ppr = new PPRCPUMTCilkRevVanilla(graph);
    }

    if (gIsDynamic)
        ppr->DynamicExecute();
    else
        ppr->Execute();

    Profiler::ReportProfile();

    return 0;
}