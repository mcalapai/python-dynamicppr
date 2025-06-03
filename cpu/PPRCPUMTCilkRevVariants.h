#ifndef __PPR_CPU_MT_CILK_REV_VARIANTS_H__
#define __PPR_CPU_MT_CILK_REV_VARIANTS_H__

#include "PPRCPUMTCilkRev.h"
#include "Meta.h"
#include "CilkUtil.h"

class PPRCPUMTCilkRevVanilla : public PPRCPUMTCilkRev
{
public:
    PPRCPUMTCilkRevVanilla(GraphVec *g) : PPRCPUMTCilkRev(g)
    {
    }
    void MainLoop(size_t phase_id = 0) override
    {
        std::vector<std::vector<IndexType>> &in_col_ind_ref = graph->in_col_ind;
        std::vector<IndexType> deg_ref = graph->deg;
#if defined(PROFILE)
        long long traverse_time = 0;
        TimeMeasurer timer;
#endif
        global_ft_count2 = 0;
        while (1)
        {
            IndexType vertex_frontier_count = global_ft_count;
            if (vertex_frontier_count == 0)
                break;
#if defined(PROFILE)
            std::cout << "iteration_id=" << iteration_id << ",frontier=" << vertex_frontier_count << std::endl;
#endif
// First parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none) \
    shared(vertex_frontier_count, global_ft, global_ft_r, residual, pagerank, ALPHA, vertex_offset, in_col_ind_ref)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                global_ft_r[i] = residual[u];
                pagerank[u] += ALPHA * residual[u];
                residual[u] = 0.0;
                vertex_offset[i] = in_col_ind_ref[u].size();
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif
// Second main parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none)                                                                      \
    shared(vertex_frontier_count, global_ft, global_ft_r, in_col_ind_ref, deg_ref, VERTEX_DEGREE_THRESHOLD, \
               vertex_offset, edge_ind, edge_flag, /*residual via AtomicAddResidual*/ ALPHA, phase_id, gTolerance, status, iteration_id, locks)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                ValueType ru = global_ft_r[i];
                IndexType indegu = in_col_ind_ref[u].size();

                if (indegu < VERTEX_DEGREE_THRESHOLD)
                {
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;
                        if (IsLegalPush(curr, phase_id))
                        {
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp)
                            {
                                is_frontier = true;
                            }
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
                else
                { // Parallel inner loop: Loop variable 'j' implicitly private
#pragma omp parallel for default(none)                                                                                                                                       \
    shared(indegu, vertex_offset, in_col_ind_ref, deg_ref, edge_ind, edge_flag, /*residual via AtomicAddResidual*/ ALPHA, phase_id, gTolerance, status, iteration_id, locks) \
    firstprivate(i, u, ru)
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;
                        if (IsLegalPush(curr, phase_id))
                        {
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp)
                            {
                                is_frontier = true;
                            }
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
            }
#if defined(PROFILE)
            Profiler::EndTimer(EXPAND_TIME);
            timer.EndTimer();
            traverse_time += timer.GetElapsedMicroSeconds();
#endif
            global_ft_count2 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);

            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }
};

class PPRCPUMTCilkRevEager : public PPRCPUMTCilkRev
{
public:
    PPRCPUMTCilkRevEager(GraphVec *g) : PPRCPUMTCilkRev(g)
    {
    }
    void MainLoop(size_t phase_id = 0) override
    {
        std::vector<std::vector<IndexType>> &in_col_ind_ref = graph->in_col_ind;
        std::vector<IndexType> deg_ref = graph->deg;
#if defined(PROFILE)
        long long traverse_time = 0;
        TimeMeasurer timer;
#endif
        global_ft_count2 = 0;
        while (1)
        {
            IndexType vertex_frontier_count = global_ft_count;
            if (vertex_frontier_count == 0)
                break;
#if defined(PROFILE)
            std::cout << "iteration_id=" << iteration_id << ",frontier=" << vertex_frontier_count << std::endl;
#endif
// First parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none) \
    shared(vertex_frontier_count, global_ft, status, iteration_id, vertex_offset, in_col_ind_ref)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                vertex_offset[i] = in_col_ind_ref[u].size();
                status[u] = iteration_id;
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif
// Second main parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none)                                                               \
    shared(vertex_frontier_count, global_ft, residual, global_ft_r, pagerank, ALPHA,                 \
               in_col_ind_ref, deg_ref, vertex_offset, VERTEX_DEGREE_THRESHOLD, edge_ind, edge_flag, \
               phase_id, gTolerance, status, iteration_id, locks)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                ValueType ru = residual[u];
                global_ft_r[i] = ru;
                pagerank[u] += ALPHA * ru;
                IndexType indegu = in_col_ind_ref[u].size();

                if (indegu < VERTEX_DEGREE_THRESHOLD)
                {
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(curr, phase_id))
                        {
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp)
                            {
                                is_frontier = true;
                            }
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
                else
                { // Parallel inner loop: Loop variable 'j' implicitly private
#pragma omp parallel for default(none)                                                                                                                                       \
    shared(indegu, vertex_offset, in_col_ind_ref, deg_ref, edge_ind, edge_flag, /*residual via AtomicAddResidual*/ ALPHA, phase_id, gTolerance, status, iteration_id, locks) \
    firstprivate(i, u, ru)
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(curr, phase_id))
                        {
                            bool resp = AtomicUpdateStatus(v, iteration_id);
                            if (resp)
                            {
                                is_frontier = true;
                            }
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
            }
#if defined(PROFILE)
            Profiler::EndTimer(EXPAND_TIME);
            timer.EndTimer();
            traverse_time += timer.GetElapsedMicroSeconds();
#endif

            IndexType new_frontier_count1 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);

            IndexType *vertex_ind_local = edge_ind;
            bool *vertex_flag_local = edge_flag;
// Third parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none) \
    shared(vertex_frontier_count, global_ft, residual, global_ft_r, phase_id, gTolerance, vertex_flag_local, vertex_ind_local)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                residual[u] -= global_ft_r[i];
                if (IsLegalPush(residual[u], phase_id))
                {
                    vertex_flag_local[i] = true;
                    vertex_ind_local[i] = u;
                }
                else
                {
                    vertex_flag_local[i] = false;
                }
            }

            IndexType new_frontier_count2 = sequence::pack<IndexType, IndexType>(vertex_ind_local, global_ft2 + new_frontier_count1, vertex_flag_local, vertex_frontier_count);
            global_ft_count2 = new_frontier_count1 + new_frontier_count2;

            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }
};
class PPRCPUMTCilkRevFF : public PPRCPUMTCilkRev
{
public:
    PPRCPUMTCilkRevFF(GraphVec *g) : PPRCPUMTCilkRev(g)
    {
    }
    void MainLoop(size_t phase_id = 0) override
    {
        std::vector<std::vector<IndexType>> &in_col_ind_ref = graph->in_col_ind;
        std::vector<IndexType> deg_ref = graph->deg;
#if defined(PROFILE)
        long long traverse_time = 0;
        TimeMeasurer timer;
#endif
        global_ft_count2 = 0;
        while (1)
        {
            IndexType vertex_frontier_count = global_ft_count;
            if (vertex_frontier_count == 0)
                break;
#if defined(PROFILE)
            std::cout << "iteration_id=" << iteration_id << ",frontier=" << vertex_frontier_count << std::endl;
#endif
// First parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none) \
    shared(vertex_frontier_count, global_ft, global_ft_r, residual, pagerank, ALPHA, vertex_offset, in_col_ind_ref)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                global_ft_r[i] = residual[u];
                pagerank[u] += ALPHA * residual[u];
                residual[u] = 0.0;
                vertex_offset[i] = in_col_ind_ref[u].size();
            }
            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(vertex_offset, vertex_offset, vertex_frontier_count);
#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif
// Second main parallel loop: Loop variable 'i' implicitly private
#pragma omp parallel for default(none)                                                                      \
    shared(vertex_frontier_count, global_ft, global_ft_r, in_col_ind_ref, deg_ref, VERTEX_DEGREE_THRESHOLD, \
               vertex_offset, edge_ind, edge_flag, /*residual via AtomicAddResidual*/ ALPHA, phase_id, gTolerance)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = global_ft[i];
                ValueType ru = global_ft_r[i];
                IndexType indegu = in_col_ind_ref[u].size();

                if (indegu < VERTEX_DEGREE_THRESHOLD)
                {
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(prer, phase_id) == false && IsLegalPush(curr, phase_id) == true)
                        {
                            is_frontier = true;
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
                else
                { // Parallel inner loop: Loop variable 'j' implicitly private
#pragma omp parallel for default(none)                                                                                                          \
    shared(indegu, vertex_offset, in_col_ind_ref, deg_ref, edge_ind, edge_flag, /*residual via AtomicAddResidual*/ ALPHA, phase_id, gTolerance) \
    firstprivate(i, u, ru)
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        ValueType add = (1.0 - ALPHA) * ru / (deg_ref[v] + 1);
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(prer, phase_id) == false && IsLegalPush(curr, phase_id) == true)
                        {
                            is_frontier = true;
                        }
                        if (is_frontier)
                        {
                            edge_ind[off] = v;
                            edge_flag[off] = true;
                        }
                        else
                        {
                            edge_flag[off] = false;
                        }
                    }
                }
            }
#if defined(PROFILE)
            Profiler::EndTimer(EXPAND_TIME);
            timer.EndTimer();
            traverse_time += timer.GetElapsedMicroSeconds();
#endif
            global_ft_count2 = sequence::pack<IndexType, IndexType>(edge_ind, global_ft2, edge_flag, total_granual);

            std::swap(global_ft, global_ft2);
            std::swap(global_ft_count, global_ft_count2);
            ++iteration_id;
        }
#if defined(PROFILE)
        std::cout << "traverse_time=" << traverse_time / 1000.0 << "ms" << std::endl;
#endif
    }
};
#endif