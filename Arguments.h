#ifndef __ARGUMENTS_H__
#define __ARGUMENTS_H__

#include <iostream>
#include <cassert>
#include "Meta.h"
#include "CommandLine.h"

static void PrintUsage()
{
    std::cout << "==========[USAGE]==========" << std::endl;
    std::cout << "-d: gDataFileName" << std::endl;
    std::cout << "-a: gAppType" << std::endl;
    std::cout << REVERSE_PUSH << ":rev push" << std::endl;
    std::cout << "-i: gIsDirected" << std::endl;
    std::cout << "-y: gIsDynamic (1 for dynamic, 0 for static)" << std::endl;

    std::cout << "--- Dynamic Graph Parameters (if -y 1) ---" << std::endl;
    std::cout << "-w: gWindowRatio" << std::endl;
    std::cout << "-n: gWorkloadConfigType" << std::endl;
    std::cout << SLIDE_WINDOW_RATIO << ": SLIDE_WINDOW_RATIO, " << SLIDE_BATCH_SIZE << ": SLIDE_BATCH_SIZE" << std::endl;
    std::cout << "-r: gStreamUpdateCountVersusWindowRatio" << std::endl;
    std::cout << "-b: gStreamBatchCount" << std::endl;
    std::cout << "-c: gStreamUpdateCountPerBatch" << std::endl;
    std::cout << "-l: gStreamUpdateCountTotal" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    std::cout << "-s: gSourceVertexId" << std::endl;
    std::cout << "-t: gThreadNum" << std::endl;
    std::cout << "-o: gVariant" << std::endl;
    std::cout << OPTIMIZED << ": optimized, " << FAST_FRONTIER << ": fast frontier, " << EAGER << ": eager, " << VANILLA << ": VANILLA" << std::endl;
    std::cout << "-e: error tolerance" << std::endl;
    std::cout << "STATIC EXAMPLE: ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 0 -s 1 -t 4" << std::endl;
    std::cout << "DYNAMIC EXAMPLE (Mode 0): ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -w 0.1 -n 0 -r 0.01 -b 100 -s 1 -t 4" << std::endl;
    std::cout << "DYNAMIC EXAMPLE (Mode 1): ./pagerank -d ../data/com-dblp.ungraph.bin -a 0 -i 0 -y 1 -w 0.1 -n 1 -c 100 -l 10000 -s 1 -t 4" << std::endl;
}
static void PrintArguments()
{
    std::cout << "gDataFileName=" << gDataFileName << std::endl;
    std::cout << "gAppType=" << gAppType << ",gIsDirected=" << gIsDirected << ",gIsDynamic=" << gIsDynamic << std::endl;
    if (gIsDynamic)
    {
        std::cout << "gWindowRatio=" << gWindowRatio << ",gWorkloadConfigType=" << gWorkloadConfigType << ",gStreamUpdateCountVersusWindowRatio=" << gStreamUpdateCountVersusWindowRatio << ",gStreamBatchCount=" << gStreamBatchCount << ",gStreamUpdateCountPerBatch=" << gStreamUpdateCountPerBatch << ",gStreamUpdateCountTotal=" << gStreamUpdateCountTotal << std::endl;
    }
    std::cout << "gSourceVertexId=" << gSourceVertexId << std::endl;
    std::cout << "gThreadNum=" << gThreadNum << ",gVariant=" << gVariant << std::endl;
    std::cout << "error=" << gTolerance << ",ALPHA=" << ALPHA << std::endl;
}

static void ArgumentsChecker()
{
    bool valid = true;
    if (gAppType < 0 || gAppType > kAlgoTypeSize)
    {
        valid = false;
    }
    if (gIsDirected < 0 || gIsDynamic < 0 || gDataFileName == "")
    { // gIsDynamic check for <0 is fine
        valid = false;
    }

    if (gIsDynamic == 1)
    { // Only check workload configs if graph is dynamic
        if (gWorkloadConfigType == SLIDE_WINDOW_RATIO)
        {
            if (gStreamUpdateCountVersusWindowRatio < 0.0 || gStreamBatchCount == 0)
                valid = false;
        }
        else if (gWorkloadConfigType == SLIDE_BATCH_SIZE)
        {
            if (gStreamUpdateCountPerBatch == 0 || gStreamUpdateCountTotal == 0)
                valid = false;
        }
        else
        { // If dynamic, one of the workload types must be specified.
            std::cout << "Invalid gWorkloadConfigType for dynamic graph." << std::endl;
            valid = false;
        }
    }
    // No specific checks for gIsDynamic == 0 regarding workload, as they don't apply.

    if (!valid)
    {
        std::cout << "Invalid arguments detected." << std::endl;
        PrintArguments(); // Print the arguments that were parsed
        PrintUsage();
        exit(-1);
    }
}

static void ArgumentsParser(int argc, char *argv[])
{
    CommandLine commandline(argc, argv);

    gDataFileName = commandline.GetOptionValue("-d", "");
    gAppType = commandline.GetOptionIntValue("-a", 0);
    gIsDirected = commandline.GetOptionIntValue("-i", -1);
    gIsDynamic = commandline.GetOptionIntValue("-y", -1); // Default to -1 to ensure it's explicitly set

    // For dynamic graphs, these have defaults or are parsed.
    // For static graphs, they are parsed but should be ignored by ArgumentsChecker if gIsDynamic is 0.
    gWindowRatio = commandline.GetOptionDoubleValue("-w", 0.1);
    gWorkloadConfigType = commandline.GetOptionIntValue("-n", SLIDE_WINDOW_RATIO);
    gStreamUpdateCountVersusWindowRatio = commandline.GetOptionDoubleValue("-r", -1.0);
    gStreamBatchCount = commandline.GetOptionIntValue("-b", 0);
    gStreamUpdateCountPerBatch = commandline.GetOptionIntValue("-c", 0);
    gStreamUpdateCountTotal = commandline.GetOptionIntValue("-l", 0);

    gSourceVertexId = commandline.GetOptionIntValue("-s", 1);
    gThreadNum = commandline.GetOptionIntValue("-t", 1);
    gVariant = commandline.GetOptionIntValue("-o", 0); // OPTIMIZED
    gTolerance = commandline.GetOptionDoubleValue("-e", 0.000000001);

    ArgumentsChecker(); // This will now correctly handle static vs dynamic
}

#endif