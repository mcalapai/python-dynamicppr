#ifndef __PPR_CPU_MT_CILK_H__
#define __PPR_CPU_MT_CILK_H__

#include "GraphVec.h"
#include "SlidingGraphVec.h"
#include "TimeMeasurer.h"
#include "Profiler.h"
#include "Barrier.h"
#include "SpinLock.h"
#include "CilkUtil.h" // Still needed for sequence namespace and other utilities
#include "PPRCPUPowVec.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <queue>
#include <map>
#include <atomic>
#include <algorithm>
#include <omp.h> // For OpenMP
#if defined(PAPI_PROFILE)
#include "PapiProfiler.h"
#endif

class PPRCPUMTCilk
{
public:
    PPRCPUMTCilk(GraphVec *g) : graph(g)
    {
        vertex_count = g->vertex_count;
        edge_count = g->edge_count;

        pagerank = new ValueType[vertex_count + 1];
        residual = new ValueType[vertex_count + 1];
        iteration_id = 0;

        source_vertex_id = gSourceVertexId;

        predeg = new IndexType[vertex_count];

        thread_count = gThreadNum;
        SetWorkers(thread_count); // This now calls omp_set_num_threads
        omp_set_nested(1);        // Enable nested parallelism

        locks = new SpinLock[vertex_count];
        for (IndexType i = 0; i < vertex_count; ++i)
            locks[i].Init();
        is_over = false;

        ppr_time = 0;
    }
    ~PPRCPUMTCilk()
    {
        delete[] pagerank;
        pagerank = NULL;
        delete[] residual;
        residual = NULL;
        delete[] predeg;
        predeg = NULL;
        delete[] locks;
        locks = NULL;
    }

    virtual void Execute()
    {
        std::cout << "start processing..." << std::endl;
        Profiler::StartTimer(TOTAL_TIME);
        TimeMeasurer timer;
        timer.StartTimer();

        ExecuteImpl();

        timer.EndTimer();
        Profiler::EndTimer(TOTAL_TIME);
        std::cout << std::endl
                  << "elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
#if defined(VALIDATE)
        Validate();
#endif
    }

    virtual void DynamicExecute()
    {
        std::cout << "start processing..." << std::endl;
#if defined(PAPI_PROFILE)
        PapiProfiler::InitPapiProfiler();
#endif
        Profiler::StartTimer(TOTAL_TIME);

        Profiler::StartTimer(INIT_GRAPH_CALC_TIME);
        TimeMeasurer timer;
        timer.StartTimer();
        ExecuteImpl();
        timer.EndTimer();
        std::cout << std::endl
                  << "elapsed time=" << timer.GetElapsedMicroSeconds() / 1000.0 << "ms" << std::endl;
        Profiler::EndTimer(INIT_GRAPH_CALC_TIME);
#if defined(VALIDATE)
        Validate();
#endif
        Profiler::StartTimer(DYNA_GRAPH_CALC_TIME);
        DynamicMainLoop();
        Profiler::EndTimer(DYNA_GRAPH_CALC_TIME);

        std::cout << "===== PageRank Results (Source: " << this->source_vertex_id << ") =====" << std::endl;
        for (IndexType i = 0; i < this->vertex_count; ++i)
        {
            // Potentially filter for non-zero or significant scores if the vector is sparse
            if (this->pagerank[i] > 0.0)
            { // Example filter
                std::cout << "Vertex: " << i << ", PageRank: " << std::fixed << std::setprecision(12) << this->pagerank[i] << std::endl;
            }
        }
        std::cout << "===== End PageRank Results =====" << std::endl;

        Profiler::EndTimer(TOTAL_TIME);
#if defined(PAPI_PROFILE)
        PapiProfiler::ReportProfile();
#endif
    }

    virtual void DynamicMainLoop()
    {
        SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec *>(graph);
        // on streaming updates
        size_t stream_batch_count = 0;
        while (stream_batch_count++ < gStreamBatchCount)
        {
            Profiler::StartTimer(EXCLUDE_GRAPH_UPDATE_TIME);
            // report throughput from time to time
            if (gStreamUpdateCountPerBatch >= 100 || stream_batch_count % 1000 == 0)
            {
                std::cout << "stream_batch_count=" << stream_batch_count << std::endl;
                double cur_ppr_time = ppr_time / 1000.0;
                long long cur_edge_count = gStreamUpdateCountPerBatch;
                cur_edge_count *= (stream_batch_count - 1);
                std::cout << "ppr_time " << cur_ppr_time << " ms" << std::endl;
                std::cout << "edge_count " << cur_edge_count << std::endl;
                std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? cur_ppr_time / (stream_batch_count - 1) : 0) << " ms" << std::endl;
                std::cout << "ppr_throughput " << cur_edge_count / cur_ppr_time * 1000.0 << " edge/s" << std::endl;
            }

            bool is_edge_stream_over = dg->StreamUpdates(gStreamUpdateCountPerBatch);
            if (is_edge_stream_over)
                break;

            // update graph structure
            dg->IncConstructWindowGraph();
            // dg->ConstructGraph();

            Profiler::EndTimer(EXCLUDE_GRAPH_UPDATE_TIME);
#if defined(PAPI_PROFILE)
            PapiProfiler::BeginProfile();
#endif
            Profiler::StartTimer(PPR_TIME);
            TimeMeasurer timer;
            timer.StartTimer();
            IncExecuteImpl();
            timer.EndTimer();
            ppr_time += timer.GetElapsedMicroSeconds();
            Profiler::EndTimer(PPR_TIME);
#if defined(PAPI_PROFILE)
            PapiProfiler::EndProfile();
#endif

#if defined(VALIDATE)
            Validate();
#endif
        }
        std::cout << "stream_batch_count=" << stream_batch_count << std::endl;
        double cur_ppr_time = ppr_time / 1000.0;
        long long cur_edge_count = gStreamUpdateCountPerBatch;
        cur_edge_count *= (stream_batch_count - 1);
        std::cout << "ppr_time " << cur_ppr_time << " ms" << std::endl;
        std::cout << "edge_count " << cur_edge_count << std::endl;
        std::cout << "ppr_latency " << ((stream_batch_count - 1 > 0) ? cur_ppr_time / (stream_batch_count - 1) : 0) << " ms" << std::endl;
        std::cout << "ppr_throughput " << cur_edge_count / cur_ppr_time * 1000.0 << " edge/s" << std::endl;
    }

    void RevertOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType edge_batch_length, IndexType *deg)
    {
#pragma omp parallel for default(none) shared(edge_batch_length, edge_batch1, is_insert, deg, locks)
        for (IndexType i = 0; i < edge_batch_length; ++i)
        {
            IndexType u = edge_batch1[i];
            locks[u].Lock();
            if (is_insert[i])
                deg[u]--;
            else
                deg[u]++;
            locks[u].Unlock();
        }
    }
    void CopyOutDegree(IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length, IndexType *predeg_arg)
    {
        // Renamed predeg to predeg_arg to avoid conflict with member predeg if this function were non-static
        // For member functions, 'this->predeg' or just 'predeg' refers to the member.
        // The argument is also named predeg in the original, which is fine.
        std::vector<IndexType> deg_ref = graph->deg; // Capture graph->deg by reference for clarity in shared clause
#pragma omp parallel for default(none) shared(edge_batch_length, edge_batch1, edge_batch2, predeg_arg, graph, deg_ref)
        for (IndexType i = 0; i < edge_batch_length; ++i)
        {
            IndexType u = edge_batch1[i];
            IndexType v = edge_batch2[i];
            predeg_arg[u] = deg_ref[u]; // graph->deg[u]
            predeg_arg[v] = deg_ref[v]; // graph->deg[v]
        }
    }

    virtual void ExecuteImpl() = 0;

    virtual void IncExecuteImpl() = 0;

    virtual void Validate() = 0;

public: // Or public, if called from outside
    void PrintNodeStructure(IndexType node_id_to_inspect)
    {
        if (!this->graph)
        {
            std::cout << "Graph object is null for structure inspection." << std::endl;
            return;
        }

        if (node_id_to_inspect >= this->graph->vertex_count)
        {
            std::cout << "Node " << node_id_to_inspect
                      << " is out of bounds (vertex_count: " << this->graph->vertex_count << ")" << std::endl;
            return;
        }

        std::cout << "\n===== Graph Structure around Node: " << node_id_to_inspect
                  << " (vertex_count: " << this->graph->vertex_count << ") =====" << std::endl;

        // Out-degree and Outgoing Neighbors
        size_t col_ind_size = 0;
        if (node_id_to_inspect < this->graph->col_ind.size())
        {
            col_ind_size = this->graph->col_ind[node_id_to_inspect].size();
        }
        std::cout << "Out-degree (from col_ind size): " << col_ind_size << std::endl;
        std::cout << "Outgoing Neighbors (Edges from " << node_id_to_inspect << " to N):" << std::endl;
        if (col_ind_size == 0)
        {
            std::cout << "  None" << std::endl;
        }
        else
        {
            for (IndexType neighbor : this->graph->col_ind[node_id_to_inspect])
            {
                std::cout << "  -> " << neighbor << std::endl;
            }
        }

        // In-degree and Incoming Neighbors
        size_t in_col_ind_size = 0;
        if (node_id_to_inspect < this->graph->in_col_ind.size())
        {
            in_col_ind_size = this->graph->in_col_ind[node_id_to_inspect].size();
        }
        std::cout << "In-degree (from in_col_ind size): " << in_col_ind_size << std::endl;
        std::cout << "Incoming Neighbors (Edges from N to " << node_id_to_inspect << "):" << std::endl;
        if (in_col_ind_size == 0)
        {
            std::cout << "  None" << std::endl;
        }
        else
        {
            for (IndexType neighbor : this->graph->in_col_ind[node_id_to_inspect])
            {
                std::cout << "  <- " << neighbor << std::endl;
            }
        }
        std::cout << "===== End Graph Structure for Node " << node_id_to_inspect << " =====" << std::endl
                  << std::endl;
    }

public:
    GraphVec *graph;
    IndexType vertex_count;
    IndexType edge_count;
    // app data
    ValueType *pagerank;
    ValueType *residual;
    IndexType iteration_id;
    // app parameter
    IndexType source_vertex_id;
    // dynamic graph
    IndexType *predeg; // This is a member variable

    // multi thread
    size_t thread_count;
    volatile bool is_over;
    SpinLock *locks;

    // measurement
    double ppr_time;
};

#endif