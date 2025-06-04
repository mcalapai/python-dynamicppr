#ifndef __PPR_CPU_MT_CILK_REV_H__
#define __PPR_CPU_MT_CILK_REV_H__

#include "PPRCPUMTCilk.h"
#include "Meta.h" // For gTolerance, ALPHA, VERTEX_DEGREE_THRESHOLD

// For sequence::pack, sequence::plusScan
// CilkUtil.h also includes omp.h
#include "CilkUtil.h"

class PPRCPUMTCilkRev : public PPRCPUMTCilk
{
public:
    PPRCPUMTCilkRev(GraphVec *g) : PPRCPUMTCilk(g)
    {
        edge_ind = new IndexType[edge_count + 1];
        edge_flag = new bool[edge_count + 1];
        vertex_offset = new IndexType[vertex_count + 1];
        status = new IndexType[vertex_count + 1];
        memset(status, -1, sizeof(IndexType) * (vertex_count + 1));

        global_ft = new IndexType[vertex_count];
        global_ft2 = new IndexType[vertex_count];
        global_ft_r = new ValueType[vertex_count];
    }
    ~PPRCPUMTCilkRev()
    {
        delete[] edge_ind;
        edge_ind = NULL;
        delete[] edge_flag;
        edge_flag = NULL;
        delete[] vertex_offset;
        vertex_offset = NULL;
        delete[] status;
        status = NULL;

        delete[] global_ft;
        global_ft = NULL;
        delete[] global_ft2;
        global_ft2 = NULL;
        delete[] global_ft_r;
        global_ft_r = NULL;
    }

    // In ./cpu/PPRCPUMTCilkRev.h
    virtual void ExecuteImpl()
    {
        Init();

        const int MAX_OUTER_ITERATIONS = 100;
        int outer_iter = 0;

        std::cout << "--- ExecuteImpl Start (Source: " << this->source_vertex_id << ", Tolerance: " << gTolerance << ") ---" << std::endl;

        for (outer_iter = 0; outer_iter < MAX_OUTER_ITERATIONS; ++outer_iter)
        {
            std::cout << "\nExecuteImpl: ===== Outer Iteration " << outer_iter << " ===== " << std::endl;

            // --- Phase 0: Process positive residuals ---
            std::cout << "ExecuteImpl: Starting MainLoop(0) (Pos Res). Session ID: " << this->iteration_id
                      << ". Initial global_ft_count: " << this->global_ft_count << std::endl;
            if (this->global_ft_count > 0)
            {
                MainLoop(0); // Process positive residuals
            }
            std::cout << "ExecuteImpl: Finished MainLoop(0). Current global_ft_count (should be 0): " << this->global_ft_count << std::endl;

            // --- Prepare for Phase 1 ---
            this->iteration_id++; // New session ID for negative residuals

            bool *temp_vertex_flags_ml1 = new bool[this->vertex_count];
            IndexType *all_vertex_indices_ml1 = new IndexType[this->vertex_count];
            IndexType neg_residual_nodes_for_ML1 = 0;

#pragma omp parallel for
            for (IndexType i = 0; i < this->vertex_count; ++i)
            {
                all_vertex_indices_ml1[i] = i;
                if (IsLegalPush(this->residual[i], 1))
                { // Check for residual < -gTolerance
                    temp_vertex_flags_ml1[i] = true;
#pragma omp atomic
                    neg_residual_nodes_for_ML1++; // Atomic increment if inside parallel loop modifying shared counter
                }
                else
                {
                    temp_vertex_flags_ml1[i] = false;
                }
            }
            // If the loop above is serial, remove #pragma omp atomic
            // If it's parallel, neg_residual_nodes_for_ML1 should be reduction(+:neg_residual_nodes_for_ML1)
            // For simplicity, let's make the counting part serial for a moment to avoid pragma complexity for this debug counter:
            // neg_residual_nodes_for_ML1 = 0;
            // for (IndexType i = 0; i < this->vertex_count; ++i) {
            //     if (temp_vertex_flags_ml1[i]) neg_residual_nodes_for_ML1++;
            // }

            this->global_ft_count = sequence::pack<IndexType, IndexType>(all_vertex_indices_ml1, this->global_ft, temp_vertex_flags_ml1, this->vertex_count);

            delete[] temp_vertex_flags_ml1;
            delete[] all_vertex_indices_ml1;

            std::cout << "ExecuteImpl: Built frontier for MainLoop(1) (Neg Res). Nodes to process: " << this->global_ft_count
                      << ". (Direct count of neg_violators before pack: " << neg_residual_nodes_for_ML1 << ")" << std::endl;

            IndexType ft_count_before_ml1 = this->global_ft_count;

            // --- Phase 1: Process negative residuals ---
            if (this->global_ft_count > 0)
            {
                std::cout << "ExecuteImpl: Starting MainLoop(1) (Neg Res). Session ID: " << this->iteration_id << std::endl;
                MainLoop(1);
            }
            std::cout << "ExecuteImpl: Finished MainLoop(1). Current global_ft_count (should be 0): " << this->global_ft_count << std::endl;

            // --- Prepare for next potential Phase 0 ---
            this->iteration_id++; // New session ID for the next positive residual processing phase

            bool *temp_vertex_flags_ml0_next = new bool[this->vertex_count];
            IndexType *all_vertex_indices_ml0_next = new IndexType[this->vertex_count];
            IndexType pos_residual_nodes_for_ML0_next = 0;

#pragma omp parallel for
            for (IndexType i = 0; i < this->vertex_count; ++i)
            {
                all_vertex_indices_ml0_next[i] = i;
                if (IsLegalPush(this->residual[i], 0))
                { // Check for residual > gTolerance
                    temp_vertex_flags_ml0_next[i] = true;
                    // #pragma omp atomic // if doing parallel reduction for pos_residual_nodes_for_ML0_next
                    // pos_residual_nodes_for_ML0_next++;
                }
                else
                {
                    temp_vertex_flags_ml0_next[i] = false;
                }
            }
            // Serial count for debug simplicity
            for (IndexType i = 0; i < this->vertex_count; ++i)
            {
                if (temp_vertex_flags_ml0_next[i])
                    pos_residual_nodes_for_ML0_next++;
            }

            this->global_ft_count = sequence::pack<IndexType, IndexType>(all_vertex_indices_ml0_next, this->global_ft, temp_vertex_flags_ml0_next, this->vertex_count);

            delete[] temp_vertex_flags_ml0_next;
            delete[] all_vertex_indices_ml0_next;

            std::cout << "ExecuteImpl: Built frontier for *next* MainLoop(0) pass. Nodes to process: " << this->global_ft_count
                      << ". (Direct count of pos_violators before pack: " << pos_residual_nodes_for_ML0_next << ")" << std::endl;

            // --- Convergence Check ---
            // If MainLoop(1) had no work (ft_count_before_ml1 == 0) AND
            // there are no new positive residuals for the next MainLoop(0) (this->global_ft_count == 0),
            // then we have converged.
            if (ft_count_before_ml1 == 0 && this->global_ft_count == 0)
            {
                std::cout << "ExecuteImpl: CONVERGED in outer_iter " << outer_iter
                          << ". No work for ML1 and no new positive residuals for next ML0." << std::endl;
                break;
            }
        }

        if (outer_iter == MAX_OUTER_ITERATIONS)
        {
            std::cout << "ExecuteImpl: Reached MAX_OUTER_ITERATIONS (" << MAX_OUTER_ITERATIONS << ")." << std::endl;
        }

        std::cout << "--- ExecuteImpl End ---" << std::endl;

        IndexType final_pos_violators = 0;
        IndexType final_neg_violators = 0;
        ValueType final_max_abs_res = 0.0;
        IndexType final_max_abs_res_node = -1;

        for (IndexType i = 0; i < this->vertex_count; ++i)
        {
            if (this->residual[i] > gTolerance)
                final_pos_violators++;
            if (this->residual[i] < -gTolerance)
                final_neg_violators++;
            if (std::fabs(this->residual[i]) > final_max_abs_res)
            {
                final_max_abs_res = std::fabs(this->residual[i]);
                final_max_abs_res_node = i;
            }
        }
        std::cout << "ExecuteImpl Final Residual Summary (before Validate call):" << std::endl;
        std::cout << "  Positive Violators (res >  " << gTolerance << "): " << final_pos_violators << std::endl;
        std::cout << "  Negative Violators (res < -" << gTolerance << "): " << final_neg_violators << std::endl;
        if (final_max_abs_res_node != -1)
        {
            std::cout << "  Max Abs Residual: " << std::fixed << std::setprecision(15) << final_max_abs_res
                      << " at node " << final_max_abs_res_node
                      << " (actual: " << this->residual[final_max_abs_res_node] << ")" << std::endl;
        }
        else
        {
            std::cout << "  All residuals appear within tolerance." << std::endl;
        }
    }

    virtual void IncExecuteImpl()
    {
        SlidingGraphVec *dg = reinterpret_cast<SlidingGraphVec *>(graph);
#if defined(PROFILE)
        Profiler::StartTimer(INC_UPDATE_TIME);
#endif
        EdgeBatch *edge_batch = dg->edge_batch;

        CopyOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->length, this->predeg);
        RevertOutDegree(edge_batch->edge1, edge_batch->edge2, edge_batch->is_insert, edge_batch->length, this->predeg);

        StreamUpdateAppData(edge_batch->edge1, edge_batch->edge2, edge_batch->is_insert, edge_batch->length);
#if defined(PROFILE)
        Profiler::EndTimer(INC_UPDATE_TIME);
        Profiler::StartTimer(PUSH_TIME);
#endif

        ++iteration_id;
        DynPushInit(0, edge_batch->edge1, edge_batch->edge2, edge_batch->length, edge_ind, edge_flag, this->predeg);
        MainLoop(0);

        ++iteration_id;
        DynPushInit(1, edge_batch->edge1, edge_batch->edge2, edge_batch->length, edge_ind, edge_flag, this->predeg);
        MainLoop(1);
#if defined(PROFILE)
        Profiler::EndTimer(PUSH_TIME);
#endif
    }

    inline bool IsLegalPush(ValueType r, size_t phase_id)
    {
        if ((phase_id == 0 && r > gTolerance) || (phase_id == 1 && r < -gTolerance))
        {
            return true;
        }
        return false;
    }

    inline ValueType AtomicAddResidual(IndexType u, ValueType add)
    {
        if (sizeof(ValueType) == 8)
        {
            volatile ValueType old_val, new_val;
            do
            {
                old_val = residual[u];
                new_val = old_val + add;
            } while (!__sync_bool_compare_and_swap((long long *)(residual + u), *((long long *)&old_val), *((long long *)&new_val)));
            return old_val;
        }
        else
        {
            std::cout << "CAS bad length" << std::endl;
            exit(-1);
        }
    }
    inline bool AtomicUpdateStatus(IndexType u, IndexType iter_id)
    {
        bool ret = false;
        locks[u].Lock();
        if (status[u] < iteration_id)
        {
            status[u] = iteration_id;
            ret = true;
        }
        locks[u].Unlock();
        return ret;
    }

    virtual void StreamUpdateAppData(IndexType *edge_batch1, IndexType *edge_batch2, bool *is_insert, IndexType edge_batch_length)
    {
#pragma omp parallel for default(none) shared(edge_batch_length, edge_batch1, edge_batch2, is_insert, locks, this->predeg, pagerank, residual, source_vertex_id, ALPHA)
        for (IndexType i = 0; i < edge_batch_length; ++i)
        {
            IndexType u = edge_batch1[i];
            IndexType v = edge_batch2[i];
            locks[u].Lock();
            ValueType add_val = (1.0 - ALPHA) * pagerank[v] - pagerank[u] - ALPHA * residual[u] + ALPHA * (source_vertex_id == u ? 1.0 : 0.0);
            if (is_insert[i])
            {
                this->predeg[u]++;
                residual[u] += add_val / (this->predeg[u] + 1) / ALPHA;
            }
            else
            {
                this->predeg[u]--;
                residual[u] -= add_val / (this->predeg[u] + 1) / ALPHA;
            }
            locks[u].Unlock();
        }
    }

    virtual void DynPushInit(const size_t phase_id, IndexType *edge_batch1, IndexType *edge_batch2, IndexType edge_batch_length, IndexType *vertex_ind_arg, bool *vertex_tag_arg, IndexType *vertex_map_arg)
    {
        IndexType *vertex_set[2] = {edge_batch1, edge_batch2};

#pragma omp parallel for default(none) shared(vertex_set, edge_batch_length, vertex_map_arg, vertex_count)
        for (IndexType i = 0; i < 2; ++i)
        {
#pragma omp parallel for default(none) shared(i, vertex_set, edge_batch_length, vertex_map_arg, vertex_count)
            for (IndexType j = 0; j < edge_batch_length; ++j)
            {
                IndexType u = vertex_set[i][j];
                vertex_map_arg[u] = vertex_count;
            }
        }

#pragma omp parallel for default(none) shared(vertex_set, edge_batch_length, vertex_map_arg, vertex_count, residual, phase_id, gTolerance, vertex_tag_arg, vertex_ind_arg)
        for (IndexType i = 0; i < 2; ++i)
        {
#pragma omp parallel for default(none) shared(i, vertex_set, edge_batch_length, vertex_map_arg, vertex_count, residual, phase_id, gTolerance, vertex_tag_arg, vertex_ind_arg)
            for (IndexType j = 0; j < edge_batch_length; ++j)
            {
                IndexType u = vertex_set[i][j];
                IndexType off = i * edge_batch_length + j;
                if (IsLegalPush(residual[u], phase_id) && vertex_map_arg[u] == vertex_count)
                {
                    bool success = __sync_bool_compare_and_swap(vertex_map_arg + u, vertex_count, off);
                    if (success)
                    {
                        vertex_tag_arg[off] = true;
                        vertex_ind_arg[off] = u;
                    }
                    else
                    {
                        vertex_tag_arg[off] = false;
                    }
                }
                else
                {
                    vertex_tag_arg[off] = false;
                }
            }
        }

        IndexType frontier_count = sequence::pack<IndexType, IndexType>(vertex_ind_arg, global_ft, vertex_tag_arg, 2 * edge_batch_length);
        global_ft_count = frontier_count;

#if defined(VALIDATE)
        memset(vertex_map_arg, -1, sizeof(IndexType) * vertex_count);
        for (IndexType i = 0; i < global_ft_count; ++i)
        {
            IndexType u_val = global_ft[i];
            assert(IsLegalPush(residual[u_val], phase_id));
            assert(vertex_map_arg[u_val] == -1);
            vertex_map_arg[u_val] = u_val;
        }
        for (IndexType i = 0; i < 2; ++i)
        {
            for (IndexType j = 0; j < edge_batch_length; ++j)
            {
                IndexType u_val = vertex_set[i][j];
                if (IsLegalPush(residual[u_val], phase_id))
                    assert(vertex_map_arg[u_val] >= 0);
            }
        }
#endif
    }

    void Init()
    {
#pragma omp parallel for default(none) shared(vertex_count, pagerank, residual, source_vertex_id)
        for (IndexType u = 0; u < vertex_count; ++u)
        {
            pagerank[u] = 0.0;
            residual[u] = (source_vertex_id == u) ? 1.0 : 0.0;
        }
        global_ft[0] = source_vertex_id;
        global_ft_count = 1;
    }

    // In ./cpu/PPRCPUMTCilkRev.h

    // Ensure this MainLoop is the one from the previous "Vanilla-like" attempt:
    virtual void MainLoop(size_t phase_id = 0)
    {
        std::vector<std::vector<IndexType>> &in_col_ind_ref = graph->in_col_ind;
        const std::vector<IndexType> deg_ref = graph->deg; // Corrected: deg_ref
#if defined(PROFILE)
        long long traverse_time = 0;
        TimeMeasurer timer;
#endif
        this->global_ft_count2 = 0;

        while (1)
        {
            IndexType vertex_frontier_count = this->global_ft_count;
            if (vertex_frontier_count == 0)
            {
                // std::cout << "  MainLoop(phase=" << phase_id << ", session_id=" << this->iteration_id << "): Frontier empty, exiting loop." << std::endl;
                break;
            }
#if defined(PROFILE) || defined(DEBUG_LOGGING) // Use a custom DEBUG_LOGGING flag if PROFILE is too verbose
                                               // This print can be very frequent. Consider conditional compilation or a flag.
                                               // std::cout << "  MainLoop(phase=" << phase_id << ", session_id=" << this->iteration_id << "): Iteration with frontier_size=" << vertex_frontier_count << std::endl;
#endif

#pragma omp parallel for default(none) \
    shared(vertex_frontier_count, global_ft, global_ft_r, residual, pagerank, ALPHA, vertex_offset, in_col_ind_ref)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = this->global_ft[i];
                ValueType current_ru = this->residual[u];
                this->global_ft_r[i] = current_ru;
                this->pagerank[u] += ALPHA * current_ru;
                this->residual[u] = 0.0;
                this->vertex_offset[i] = in_col_ind_ref[u].size();
            }

            IndexType total_granual = sequence::plusScan<IndexType, IndexType>(this->vertex_offset, this->vertex_offset, vertex_frontier_count);

#if defined(PROFILE)
            timer.StartTimer();
            Profiler::StartTimer(EXPAND_TIME);
#endif

#pragma omp parallel for default(none)                                                                         \
    shared(vertex_frontier_count, global_ft, global_ft_r, residual, ALPHA,                                     \
               in_col_ind_ref, deg_ref, vertex_offset, VERTEX_DEGREE_THRESHOLD, edge_ind, edge_flag, phase_id, \
               gTolerance, status, iteration_id, locks)
            for (IndexType i = 0; i < vertex_frontier_count; ++i)
            {
                IndexType u = this->global_ft[i];            // u is the node from the current frontier
                ValueType ru_to_push = this->global_ft_r[i]; // residual of u that was captured
                IndexType indegu = in_col_ind_ref[u].size(); // in-degree of u

                if (indegu < VERTEX_DEGREE_THRESHOLD)
                {
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        IndexType off = this->vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j]; // v is an in-neighbor of u (v -> u)
                        bool is_frontier = false;
                        IndexType current_deg_v = deg_ref[v]; // out-degree of v
                        IndexType denominator = (current_deg_v == 0) ? 1 : current_deg_v;
                        ValueType add = (1.0 - ALPHA) * ru_to_push / denominator;
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(curr, phase_id))
                        {
                            if (AtomicUpdateStatus(v, this->iteration_id))
                            {
                                is_frontier = true;
                            }
                        }
                        this->edge_flag[off] = is_frontier;
                        if (is_frontier)
                            this->edge_ind[off] = v;
                        else
                            this->edge_ind[off] = 0; // Ensure valid value if not frontier
                    }
                }
                else
                {
#pragma omp parallel for default(none)                                                                                                                                                       \
    shared(indegu, u, /*ru_to_push is needed*/ vertex_offset, in_col_ind_ref, deg_ref, edge_ind, edge_flag, residual, phase_id, gTolerance, ALPHA, status, iteration_id, locks, global_ft_r) \
    firstprivate(i) // i is the index for u in global_ft and global_ft_r
                    for (IndexType j = 0; j < indegu; ++j)
                    {
                        // Note: The 'u' in shared refers to the outer loop's u. This is correct.
                        // ru_to_push should be taken from global_ft_r[i] for the specific outer 'i'
                        ValueType ru_for_inner_push = global_ft_r[i];
                        IndexType off = this->vertex_offset[i] + j;
                        IndexType v = in_col_ind_ref[u][j];
                        bool is_frontier = false;
                        IndexType current_deg_v = deg_ref[v];
                        IndexType denominator = (current_deg_v == 0) ? 1 : current_deg_v;
                        ValueType add = (1.0 - ALPHA) * ru_for_inner_push / denominator;
                        ValueType prer = AtomicAddResidual(v, add);
                        ValueType curr = prer + add;

                        if (IsLegalPush(curr, phase_id))
                        {
                            if (AtomicUpdateStatus(v, this->iteration_id))
                            {
                                is_frontier = true;
                            }
                        }
                        this->edge_flag[off] = is_frontier;
                        if (is_frontier)
                            this->edge_ind[off] = v;
                        else
                            this->edge_ind[off] = 0;
                    }
                }
            }
#if defined(PROFILE)
            Profiler::EndTimer(EXPAND_TIME);
            timer.EndTimer();
            // traverse_time += timer.GetElapsedMicroSeconds(); // traverse_time is local, careful with parallel accumulation
#endif
            IndexType new_frontier_count = sequence::pack<IndexType, IndexType>(this->edge_ind, this->global_ft2, this->edge_flag, total_granual);
            this->global_ft_count2 = new_frontier_count;

            std::swap(this->global_ft, this->global_ft2);
            std::swap(this->global_ft_count, this->global_ft_count2);
        }
        // #if defined(PROFILE) || defined(DEBUG_LOGGING)
        //     std::cout << "  MainLoop(phase=" << phase_id << ", session_id=" << this->iteration_id << ") finished." << std::endl;
        // #endif
    }

    // In class PPRCPUMTCilkRev
    virtual void Validate()
    {
        std::cout << "--- PPRCPUMTCilkRev::Validate() Start ---" << std::endl;
        graph->ConstructGraph(); // Reconstructs graph to ensure current state if dynamic, or just re-asserts if static

        IndexType positive_violators = 0;
        IndexType negative_violators = 0;
        ValueType max_abs_residual = 0.0;
        IndexType max_abs_residual_node = -1;
        ValueType min_residual_seen = 1.0;  // Assuming residuals are mostly positive or small negative
        ValueType max_residual_seen = -1.0; // Assuming residuals are mostly positive or small negative

        const int MAX_VIOLATORS_TO_PRINT = 20; // Limit how many violating nodes we print details for
        int printed_violators_count = 0;

        std::cout << std::fixed << std::setprecision(15); // For printing ValueType

        for (IndexType u = 0; u < vertex_count; ++u)
        {
            if (residual[u] < -gTolerance || residual[u] > gTolerance)
            {
                if (residual[u] > gTolerance)
                    positive_violators++;
                if (residual[u] < -gTolerance)
                    negative_violators++;

                if (std::fabs(residual[u]) > std::fabs(max_abs_residual))
                {                                   // Use fabs for comparison if max_abs_residual could be negative init
                    max_abs_residual = residual[u]; // Store actual residual, not abs
                    max_abs_residual_node = u;
                }

                if (printed_violators_count < MAX_VIOLATORS_TO_PRINT)
                {
                    std::cout << "  VALIDATE_FAIL: Node " << u
                              << ", Residual: " << residual[u]
                              << ", PageRank: " << pagerank[u]
                              << ", OutDegree: " << graph->deg[u]
                              << ", InDegree: " << graph->in_col_ind[u].size()
                              << (u == source_vertex_id ? " (SOURCE_VERTEX)" : "")
                              << std::endl;
                    // Optional: Print in-neighbors and their residuals if helpful (can be very verbose)
                    // if (graph->in_col_ind[u].size() < 10) { // Only for nodes with few in-neighbors
                    //     for (IndexType neighbor_v : graph->in_col_ind[u]) {
                    //         std::cout << "    In-Neighbor " << neighbor_v << " Res: " << residual[neighbor_v] << std::endl;
                    //     }
                    // }
                    printed_violators_count++;
                }
            }
            if (residual[u] < min_residual_seen)
                min_residual_seen = residual[u];
            if (residual[u] > max_residual_seen)
                max_residual_seen = residual[u];
        }

        std::cout << "Validation Summary (Tolerance: " << gTolerance << "):" << std::endl;
        std::cout << "  Positive Violators (residual >  tolerance): " << positive_violators << std::endl;
        std::cout << "  Negative Violators (residual < -tolerance): " << negative_violators << std::endl;
        std::cout << "  Overall Min Residual Seen: " << min_residual_seen << std::endl;
        std::cout << "  Overall Max Residual Seen: " << max_residual_seen << std::endl;
        if (max_abs_residual_node != -1)
        {
            std::cout << "  Node with Max Abs Residual: " << max_abs_residual_node
                      << ", Value: " << max_abs_residual << std::endl;
        }
        else if (positive_violators == 0 && negative_violators == 0)
        {
            std::cout << "  All residuals appear within tolerance." << std::endl;
        }

        // Original assertion and power iteration validation (if still desired after debugging)
        if (positive_violators > 0 || negative_violators > 0)
        {
            std::cout << "ERROR: Residuals out of tolerance found. See details above." << std::endl;
            // The assertion will now happen, but we have logs.
        }
        else
        {
            std::cout << "SUCCESS: All residuals within tolerance." << std::endl;
        }

        // The Power Vector validation part (optional, can be very slow for large graphs)
        // Consider enabling this only after the primary residual tolerance issue is fixed.
#if 0 // Disabled for now to focus on residual tolerance
    std::cout << "Performing Power Iteration validation..." << std::endl;
    PPRCPUPowVec *ppr_pow = new PPRCPUPowVec(graph);
    ppr_pow->CalPPRRev(source_vertex_id); // This is for Reverse Push
    double *ans = ppr_pow->pagerank;
    const double bound_pow_iter = gTolerance * 100.0; // Allow larger bound for comparing to power iteration
    IndexType pow_iter_mismatches = 0;

    for (IndexType u = 0; u < vertex_count; ++u)
    {
        ValueType err = ans[u] - pagerank[u];
        if (std::fabs(err) > bound_pow_iter) {
            if (pow_iter_mismatches < MAX_VIOLATORS_TO_PRINT) {
                std::cout << "  VALIDATE_POW_FAIL: Node " << u
                          << ", AlgoPR: " << pagerank[u]
                          << ", PowIterPR: " << ans[u]
                          << ", Diff: " << err << std::endl;
            }
            pow_iter_mismatches++;
        }
    }
    if (pow_iter_mismatches > 0) {
        std::cout << "ERROR: Mismatches found against Power Iteration: " << pow_iter_mismatches << std::endl;
    } else {
        std::cout << "SUCCESS: PageRank values match Power Iteration results within bound " << bound_pow_iter << "." << std::endl;
    }
    delete ppr_pow;
    ppr_pow = NULL;
#endif

        std::cout << "--- PPRCPUMTCilkRev::Validate() End ---" << std::endl;

        // The actual assertion that causes abort:
        // We check again here so that if the above logs are suppressed, the abort still happens.
        for (IndexType u = 0; u < vertex_count; ++u)
        {
            assert(residual[u] < gTolerance && residual[u] > -gTolerance);
        }
    }

public:
    IndexType *edge_ind;
    bool *edge_flag;
    IndexType *vertex_offset;
    IndexType *status;

    IndexType *global_ft;
    IndexType global_ft_count;
    IndexType *global_ft2;
    IndexType global_ft_count2;
    ValueType *global_ft_r;
};
#endif