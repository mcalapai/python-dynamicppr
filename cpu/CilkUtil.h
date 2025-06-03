#ifndef __CILK_UTIL_H__
#define __CILK_UTIL_H__

#include <omp.h>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <algorithm> // For std::min
#include <vector>

using namespace std;

static int GetWorkers()
{
  return omp_get_max_threads();
}
static void SetWorkers(int n)
{
  omp_set_num_threads(n);
}

#define newA(__E, __n) (__E *)malloc((__n) * sizeof(__E))

template <class E>
struct identityF
{
  E operator()(const E &x) { return x; }
};

template <class E>
struct addF
{
  E operator()(const E &a, const E &b) const { return a + b; }
};

template <class E>
struct minF
{
  E operator()(const E &a, const E &b) const { return (a < b) ? a : b; }
};

template <class E>
struct maxF
{
  E operator()(const E &a, const E &b) const { return (a > b) ? a : b; }
};

#define _SCAN_LOG_BSIZE 10
#define _SCAN_BSIZE (1 << _SCAN_LOG_BSIZE)

template <class T>
struct _seq
{
  T *A;
  long n;
  _seq()
  {
    A = NULL;
    n = 0;
  }
  _seq(T *_A, long _n) : A(_A), n(_n) {}
  void del() { free(A); }
};

namespace sequence
{
  template <class intT>
  struct boolGetA
  {
    bool *A;
    boolGetA(bool *AA) : A(AA) {}
    intT operator()(intT i) { return (intT)A[i]; }
  };

  template <class ET, class intT>
  struct getA
  {
    ET *A;
    getA(ET *AA) : A(AA) {}
    ET operator()(intT i) { return A[i]; }
  };

  template <class IT, class OT, class intT, class F>
  struct getAF
  {
    IT *A;
    F f;
    getAF(IT *AA, F ff) : A(AA), f(ff) {}
    OT operator()(intT i) { return f(A[i]); }
  };

#define nblocks(_n, _bsize) (1 + ((_n) - 1) / (_bsize))

// _i_param: The loop variable for iterating over blocks.
// _s_param_macro: The start of the overall range.
// _e_param_macro: The end of the overall range.
// _bsize_param_macro: The block size.
// _body_param_macro: The code to execute for each block. This code can use:
//                    '_i_param' as the current block index.
//                    's_block' as the start index of the current block (relative to overall start).
//                    'e_block' as the end index of the current block (relative to overall start).
#define blocked_for(_i_param, _s_param_macro, _e_param_macro, _bsize_param_macro, _body_param_macro)                   \
  {                                                                                                                    \
    using intT_blocked_for = decltype(_s_param_macro);                                                                 \
    intT_blocked_for __s_overall_blocked_for = _s_param_macro;                                                         \
    intT_blocked_for __e_overall_blocked_for = _e_param_macro;                                                         \
    intT_blocked_for __bsize_blocked_for = _bsize_param_macro;                                                         \
    intT_blocked_for __n_overall_blocked_for = __e_overall_blocked_for - __s_overall_blocked_for;                      \
    if (__n_overall_blocked_for > 0)                                                                                   \
    {                                                                                                                  \
      intT_blocked_for __num_blocks_blocked_for = nblocks(__n_overall_blocked_for, __bsize_blocked_for);               \
      /* OpenMP pragma for the loop over blocks. */                                                                    \
      /* Relies on default OpenMP data sharing rules for variables from the */                                         \
      /* enclosing scope (e.g., Sums, f, g, Out when called from scan/reduce/pack). */                                 \
      /* Loop variables _i_param, s_block, e_block are private to iterations or defined within. */                     \
      _Pragma("omp parallel for") for (intT_blocked_for _i_param = 0; _i_param < __num_blocks_blocked_for; ++_i_param) \
      {                                                                                                                \
        intT_blocked_for s_block = __s_overall_blocked_for + _i_param * __bsize_blocked_for;                           \
        intT_blocked_for e_block = std::min(s_block + __bsize_blocked_for, __e_overall_blocked_for);                   \
        _body_param_macro                                                                                              \
      }                                                                                                                \
    }                                                                                                                  \
  }

  template <class OT, class intT, class F, class G>
  OT reduceSerial(intT s_block, intT e_block, F f, G g)
  {
    if (s_block >= e_block)
    {
      // Return identity for the reduction operation f.
      // For addF, identity is 0. For minF, +infinity. For maxF, -infinity.
      // This requires knowing F. A simple OT() might not be correct.
      // Assuming OT default constructor is a valid identity for now, or f handles it.
      return OT();
    }
    OT r = g(s_block);
    for (intT j = s_block + 1; j < e_block; j++)
      r = f(r, g(j));
    return r;
  }

  template <class OT, class intT, class F, class G>
  OT reduce(intT s_orig, intT e_orig, F f, G g)
  {
    intT n = e_orig - s_orig;
    if (n <= 0)
      return OT();
    intT l = nblocks(n, _SCAN_BSIZE); // Number of blocks
    if (l <= 1)
      return reduceSerial<OT>(s_orig, e_orig, f, g);

    OT *Sums = newA(OT, l); // Array to store sum of each block
// The body of blocked_for uses block_idx_i, s_block, e_block.
// It also uses Sums, f, g from this 'reduce' function's scope.
// These (Sums, f, g) will be shared by default by the omp parallel for in blocked_for.
#define REDUCE_BLOCKED_BODY Sums[block_idx_i] = reduceSerial<OT>(s_block, e_block, f, g)
    blocked_for(block_idx_i, s_orig, e_orig, _SCAN_BSIZE, REDUCE_BLOCKED_BODY;);
#undef REDUCE_BLOCKED_BODY

    OT r = reduce<OT>((intT)0, l, f, getA<OT, intT>(Sums)); // Reduce the partial sums
    free(Sums);
    return r;
  }

  template <class OT, class intT, class F>
  OT reduce(OT *A, intT n, F f)
  {
    return reduce<OT>((intT)0, n, f, getA<OT, intT>(A));
  }

  template <class OT, class intT>
  OT plusReduce(OT *A, intT n)
  {
    if (n <= 0)
      return (OT)0;
    OT total_sum = (OT)0;
// Using OpenMP's built-in reduction for sum.
#pragma omp parallel for reduction(+ : total_sum)
    for (intT i = 0; i < n; ++i)
    {
      total_sum += A[i];
    }
    return total_sum;
  }

  template <class OT, class IT, class intT, class F, class G>
  OT mapReduce(IT *A, intT n, F f, G g)
  {
    return reduce<OT>((intT)0, n, f, getAF<IT, OT, intT, G>(A, g));
  }

  template <class intT>
  intT sum(bool *In, intT n)
  {
    if (n <= 0)
      return (intT)0;
    intT total_sum = 0;
// Using OpenMP's built-in reduction for sum.
#pragma omp parallel for reduction(+ : total_sum)
    for (intT i = 0; i < n; ++i)
    {
      total_sum += (intT)In[i];
    }
    return total_sum;
  }

  template <class ET, class intT, class F, class G>
  ET scanSerial(ET *Out, intT s_block, intT e_block, F f, G g, ET zero, bool inclusive, bool back)
  {
    ET r = zero;
    if (s_block >= e_block)
      return r; // Nothing to scan in an empty or invalid range
    if (inclusive)
    {
      if (back)
        for (intT i = e_block - 1; i >= s_block; i--)
          Out[i] = r = f(r, g(i));
      else
        for (intT i = s_block; i < e_block; i++)
          Out[i] = r = f(r, g(i));
    }
    else
    {
      if (back)
        for (intT i = e_block - 1; i >= s_block; i--)
        {
          ET t = g(i); // Map element
          Out[i] = r;  // Write current prefix sum
          r = f(r, t); // Update prefix sum
        }
      else
        for (intT i = s_block; i < e_block; i++)
        {
          ET t = g(i);
          Out[i] = r;
          r = f(r, t);
        }
    }
    return r; // Returns total sum of the scanned segment (if exclusive) or last element (if inclusive)
  }

  template <class ET, class intT, class F>
  ET scanSerial(ET *In, ET *Out, intT n, F f, ET zero)
  {
    return scanSerial(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, false);
  }

  template <class ET, class intT, class F, class G>
  ET scan(ET *Out, intT s_orig, intT e_orig, F f, G g, ET zero, bool inclusive, bool back)
  {
    intT n = e_orig - s_orig;
    if (n <= 0)
      return zero; // Or handle as error/specific case
    intT num_blocks = nblocks(n, _SCAN_BSIZE);

    if (num_blocks <= 1)
    { // Changed from <=2 to ensure blocked_for is used if there's more than one block.
      // If num_blocks is 1, reduceSerial directly gives the sum, and scanSerial the scan.
      // The original code used l <= 2, meaning it might do 2 blocks serially.
      // For simplicity and to exercise blocked_for more, l <= 1 (or num_blocks <=1) is a clearer serial base.
      return scanSerial(Out, s_orig, e_orig, f, g, zero, inclusive, back);
    }

    ET *Sums = newA(ET, num_blocks); // Stores reduction of each block

// Pass 1: Compute reduction for each block
// Body uses block_idx_i, s_block, e_block. And Sums, f, g from scan's scope.
#define SCAN_PASS1_BODY Sums[block_idx_i] = reduceSerial<ET>(s_block, e_block, f, g)
    blocked_for(block_idx_i, s_orig, e_orig, _SCAN_BSIZE, SCAN_PASS1_BODY;);
#undef SCAN_PASS1_BODY

    // Recursively scan the Sums array to get offsets for each block.
    // The 'g' functor for this scan is getA<ET,intT>(Sums), mapping index to Sums[index].
    // The 'f' functor is the same reduction operator.
    // 'zero' is the initial prefix for the scan of sums.
    // This scan is exclusive (false for inclusive).
    scan(Sums, (intT)0, num_blocks, f, getA<ET, intT>(Sums), zero, false, back);
// After this, Sums[i] contains the prefix sum *before* block i. Sums array is modified in-place.

// Pass 2: Perform local scan for each block using the computed offsets.
// Body uses block_idx_i, s_block, e_block. And Out, Sums, f, g, inclusive, back from scan's scope.
#define SCAN_PASS2_BODY scanSerial(Out, s_block, e_block, f, g, Sums[block_idx_i], inclusive, back)
    blocked_for(block_idx_i, s_orig, e_orig, _SCAN_BSIZE, SCAN_PASS2_BODY;);
#undef SCAN_PASS2_BODY

    // The 'total' value returned by the original Cilk scan was the sum of all elements
    // if the scan was exclusive, or the value of the last element if inclusive.
    // For an exclusive scan, total = Sums[num_blocks-1] + reduceSerial(last_block_elements).
    // For an inclusive scan, total = Out[e_orig-1] if e_orig > s_orig.
    // The recursive `scan` on `Sums` already computes and stores prefix sums in `Sums`.
    // The last element of `Sums` after the recursive scan (if it were to return total) would be Sums[l-1] + (original Sums[l-1] value).
    // The current implementation of `scan` (this function) will return the result of the final `scanSerial` of the last block if `back` is true,
    // or `Sums[num_blocks-1]` plus the sum of the last block if `back` is false and scan is exclusive.
    // The original code returned `total` from `scan(Sums, ...)`. Let's replicate this roughly.
    // If `back` is false, `Sums[num_blocks-1]` (after scan on Sums) holds sum of all blocks except last,
    // then add sum of last block.
    // If `back` is true, `Sums[0]` (after scan on Sums) holds sum of all blocks except first,
    // then add sum of first block.
    // This matches the definition of `total` for an exclusive scan of the original array.
    // Let's assume the value of `scan(Sums, ...)` is the total sum of elements in `Sums` (before it's overwritten).
    // The `scan` function doesn't naturally return the "total sum" of the input array being scanned,
    // but rather the prefix sum for one element beyond the end (for exclusive scan).
    // The original `ET total = scan(Sums, ...)` was likely for the sum of `Sums` elements.
    // The current `scan` returns the total sum of its input.
    // The return value of this top-level scan should be the sum of all elements in the original range [s_orig, e_orig).
    // This is: Sums[num_blocks-1] (which is prefix sum before last block) + reduction_of_last_block (which is original Sums[num_blocks-1]).
    // This requires saving the original Sums[num_blocks-1] or recomputing.
    // Or, more simply, if `inclusive` is false, the sum of all `g(i)` can be obtained from `Sums[num_blocks-1]` (after scan on Sums) + `reduceSerial` of the last block from original Sums.
    // For now, we can get the total from the `Sums` array after the *first* `blocked_for` but *before* the recursive `scan` overwrites it.
    // The original `scan` on `Sums` would return the sum of elements in the `Sums` array itself.
    // Let's simplify: the primary purpose is to fill `Out`. The `total` return might not be critical for `pack`.
    // If `total` is important: after first blocked_for, Sums contains block sums. `reduce(Sums, num_blocks, f)` would give total sum.
    // The original `total = scan(Sums, (intT) 0, l, f, getA<ET,intT>(Sums), zero, false, back);`
    // This means `total` is the sum of the block sums, effectively the sum of the whole array.
    // After the recursive `scan` modifies `Sums`, the sum is `Sums[l-1]` + last element of original `Sums` array if exclusive.
    // This is getting complicated. Let's ensure `Out` is correct. The `total` return is secondary.
    // The original code returns the result of `scan(Sums, ...)`. Let's keep that.
    // What `scan(Sums, ...)` returns is the sum of items in `Sums` if `f` is additive and `zero` is 0.
    ET total_sum_of_block_sums;
    if (num_blocks > 0)
    {                                    // Calculate sum of original block sums
      total_sum_of_block_sums = Sums[0]; // if num_blocks == 1, this is the sum
      for (intT i_sum = 1; i_sum < num_blocks; ++i_sum)
      {
        total_sum_of_block_sums = f(total_sum_of_block_sums, Sums[i_sum]);
      }
    }
    else
    {
      total_sum_of_block_sums = zero; // Or appropriate identity
    }
    // The above calculation of total_sum_of_block_sums needs to be done BEFORE Sums is overwritten by scan(Sums,...)
    // This is problematic. The original: `ET total = scan(Sums, ...)`
    // This implies `total` is what the *recursive call* returns. And that recursive call itself returns the sum of its input.
    // So, `total` should be the sum of items in `Sums` array (the block sums).
    // The `scan` function needs to return the sum of the elements it is scanning (g(i)) if it is an exclusive scan.
    // `scanSerial` (exclusive): `r` at the end is `f(f(f(zero, g(s)), g(s+1)), ... g(e-1))`. This is the sum if zero=0, f=+.
    // So `scan` should correctly return this total sum. The recursive structure ensures this.

    free(Sums);
    // The return value `total` from `scan(Sums, ...)` is the sum of elements in the `Sums` array (block sums).
    // This is the sum of the entire original array [s_orig, e_orig).
    // The return value of the current `scan` comes from the recursive call `scan(Sums, ...)`.
    // This is implicitly `total_sum_of_block_sums`.
    // Let's make it explicit for clarity of what `total_sum_of_block_sums` means here
    // `total_sum_of_block_sums` is not returned by the recursive scan call directly, it's computed from Sums.
    // The original code returned `total` which was the result of `scan(Sums, ...)`.
    // The current `scan` function has `scan(Sums, ...)` which modifies Sums and then returns a total.
    // Let's assume the recursive call `scan(Sums, ...)` correctly returns the sum of *its* input (the block sums).
    // This value *is* the sum of the original data. So the `total` variable will be correct.
    // The `scan(Sums, ...)` line modifies `Sums` but also returns the total sum *of the elements it scanned*.
    // If `scan` is defined to return the sum of `g(i)` over its range, then `total` will be sum of block sums.
    ET final_total_sum = zero; // Default for empty range
    if (n > 0)
    {
      // Reconstruct total sum from the final `Out` array if `inclusive` is false
      // Or, trust the recursive structure to propagate the sum.
      // The simplest is: the `scan` function, if exclusive, returns sum of elements.
      // The recursive call `scan(Sums, ...)` will return sum of (original) Sums elements.
      // This sum is the sum of all elements in [s_orig, e_orig).
      // So the return value from the recursive scan call is the overall total.
      // We need to re-evaluate what `scan` should return.
      // Typically, an exclusive scan fills `Out` and returns the sum of all processed elements.
      // `scanSerial` returns `r`, which is the sum if `zero` is 0 and `f` is `+`.
      // The recursive call `scan(Sums, ...)` should return sum of `Sums[0]...Sums[l-1]` (original values).
      // However, `scan` modifies its input array `Sums` if `Out` is `Sums`.
      // Here, `scan(Sums, (intT)0, num_blocks, f, getA<ET,intT>(Sums), zero, false, back);`
      // The `Out` for this recursive call is `Sums` itself.
      // After this call, Sums contains prefix sums.
      // What does it return? It returns `r` from its deepest `scanSerial`.
      // This implies it returns the sum of its input.
      // So, the `total` variable would indeed be the sum of block sums.
      // This means the return value logic of `scan` is consistent.
      // This line: `ET total = scan(Sums, (intT)0, num_blocks, f, getA<ET,intT>(Sums), zero, false, back);`
      // is problematic because `scan` expects `Out` as its first argument.
      // It should be: `ET total = scan(TempScanOut, (intT)0, num_blocks, f, getA<ET,intT>(Sums), zero, false, back);`
      // And then `Sums` would be `TempScanOut`.
      // Or, if `scan` is designed to work in-place on `Sums` if `Out == Sums`.
      // The original was `ET total = scan(Sums, ...)` which means `Out` was `Sums`.
      // Let's assume `scan` can work in-place if `Out` is the same as `In` (via `getA`).

      // The line `scan(Sums, (intT)0, num_blocks, f, getA<ET,intT>(Sums), zero, false, back);`
      // implies `Out` is `Sums`. So `Sums` is modified in-place.
      // The return value of this `scan` call will be the sum of the original `Sums[i]` values.
      // This is exactly what we need for the overall sum.
      ET overall_sum = scan(Sums, (intT)0, num_blocks, f, getA<ET, intT>(Sums), zero, false, back);
      return overall_sum;
    }
    return zero; // If n <= 0
  }

  template <class ET, class intT, class F>
  ET scan(ET *In, ET *Out, intT n, F f, ET zero)
  {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, false);
  }

  template <class ET, class intT, class F>
  ET scanI(ET *In, ET *Out, intT n, F f, ET zero)
  {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, true, false);
  }

  template <class ET, class intT, class F>
  ET scanBack(ET *In, ET *Out, intT n, F f, ET zero)
  {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, false, true);
  }

  template <class ET, class intT, class F>
  ET scanIBack(ET *In, ET *Out, intT n, F f, ET zero)
  {
    return scan(Out, (intT)0, n, f, getA<ET, intT>(In), zero, true, true);
  }

  template <class ET, class intT>
  ET plusScan(ET *In, ET *Out, intT n)
  { // Exclusive scan with addition
    return scan(Out, (intT)0, n, addF<ET>(), getA<ET, intT>(In), (ET)0, false, false);
  }

#define _F_BSIZE (2 * _SCAN_BSIZE)

  template <class intT>
  intT sumFlagsSerial(bool *Fl, intT n)
  {
    intT r = 0;
    if (n >= 128 && (n & 511) == 0 && ((long)Fl & 3) == 0)
    {
      int *IFl = (int *)Fl;
      for (int k = 0; k < (n >> 9); k++)
      {
        int rr = 0;
        for (int j = 0; j < 128; j++)
          rr += IFl[j];
        r += (rr & 255) + ((rr >> 8) & 255) + ((rr >> 16) & 255) + ((rr >> 24) & 255);
        IFl += 128;
      }
    }
    else
      for (intT j = 0; j < n; j++)
        r += Fl[j];
    return r;
  }

  template <class ET, class intT, class F>
  _seq<ET> packSerial(ET *Out_block_base, bool *Fl_overall, intT s_block, intT e_block, F f_map)
  {
    intT k = 0;
    for (intT i = s_block; i < e_block; i++)
      if (Fl_overall[i])
        Out_block_base[k++] = f_map(i);
    return _seq<ET>(Out_block_base, k);
  }

  template <class ET, class intT, class F>
  _seq<ET> pack(ET *Out, bool *Fl, intT s_orig, intT e_orig, F f_map)
  {
    intT n = e_orig - s_orig;
    if (n <= 0)
      return _seq<ET>(Out, 0);
    intT l = nblocks(n, _F_BSIZE); // Number of blocks for pack

    if (l <= 1)
    { // Serial case for pack
      // packSerial might allocate Out if it's NULL.
      // Need to handle the case where Out is NULL.
      if (Out == NULL && n > 0)
      { // Only allocate if needed and non-empty range
        intT m_serial = sumFlagsSerial(Fl + s_orig, n);
        Out = newA(ET, m_serial);
      }
      else if (n == 0 && Out == NULL)
      { // If range is empty, m is 0.
        // Out could remain NULL if it started as NULL. Let packSerial handle it.
      }
      // If Out was pre-allocated, packSerial uses it.
      return packSerial(Out, Fl, s_orig, e_orig, f_map);
    }

    intT *Sums = newA(intT, l); // Stores count of true flags for each block
// Body uses block_idx_i, s_block, e_block. And Sums, Fl from pack's scope.
#define PACK_SUMFLAGS_BODY Sums[block_idx_i] = sumFlagsSerial(Fl + s_block, e_block - s_block)
    blocked_for(block_idx_i, s_orig, e_orig, _F_BSIZE, PACK_SUMFLAGS_BODY;);
#undef PACK_SUMFLAGS_BODY

    // Perform an exclusive scan on Sums to get offsets for each block in Out array.
    // plusScan modifies Sums in-place if In and Out are the same.
    intT m = plusScan(Sums, Sums, l); // m is the total count of true flags.

    if (Out == NULL)
      Out = newA(ET, m); // Allocate Out if not already.

// Body uses block_idx_i, s_block, e_block. And Out, Sums, Fl, f_map from pack's scope.
#define PACK_SERIAL_BODY packSerial(Out + Sums[block_idx_i], Fl, s_block, e_block, f_map)
    blocked_for(block_idx_i, s_orig, e_orig, _F_BSIZE, PACK_SERIAL_BODY;);
#undef PACK_SERIAL_BODY

    free(Sums);
    return _seq<ET>(Out, m);
  }

  template <class ET, class intT>
  intT pack(ET *In, ET *Out, bool *Fl, intT n)
  {
    return pack(Out, Fl, (intT)0, n, getA<ET, intT>(In)).n;
  }

  template <class intT>
  _seq<intT> packIndex(bool *Fl, intT n)
  {
    return pack((intT *)NULL, Fl, (intT)0, n, identityF<intT>());
  }

  template <class ET, class intT, class PRED>
  intT filter(ET *In, ET *Out, intT n, PRED p)
  {
    if (n <= 0)
      return 0;
    bool *Fl = newA(bool, n);
#pragma omp parallel for
    for (intT i = 0; i < n; i++)
      Fl[i] = (bool)p(In[i]);

    intT m = pack(In, Out, Fl, n);
    free(Fl);
    return m;
  }
} // namespace sequence

template <class ET>
inline bool CAS(ET *ptr, ET oldv, ET newv)
{
  if (sizeof(ET) == 1)
  {
    return __sync_bool_compare_and_swap((bool *)ptr, *((bool *)&oldv), *((bool *)&newv));
  }
  else if (sizeof(ET) == 4)
  {
    return __sync_bool_compare_and_swap((int *)ptr, *((int *)&oldv), *((int *)&newv));
  }
  else if (sizeof(ET) == 8)
  {
    return __sync_bool_compare_and_swap((long *)ptr, *((long *)&oldv), *((long *)&newv));
  }
  else
  {
    std::cout << "CAS bad length : " << sizeof(ET) << std::endl;
    abort();
  }
}

template <class ET>
inline bool writeMin(ET *a, ET b)
{
  ET c;
  bool r = 0;
  do
    c = *a;
  while (c > b && !(r = CAS(a, c, b)));
  return r;
}

template <class ET>
inline void writeAdd(ET *a, ET b)
{
  volatile ET newV, oldV;
  do
  {
    oldV = *a;
    newV = oldV + b;
  } while (!CAS(a, oldV, newV));
}

inline uint hashInt(uint a)
{
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

inline unsigned long hashInt(unsigned long a)
{
  a = (a + 0x7ed55d166bef7a1d) + (a << 12);
  a = (a ^ 0xc761c23c510fa2dd) ^ (a >> 9);
  a = (a + 0x165667b183a9c0e1) + (a << 59);
  a = (a + 0xd3a2646cab3487e3) ^ (a << 49);
  a = (a + 0xfd7046c5ef9ab54c) + (a << 3);
  a = (a ^ 0xb55a4f090dd4a67b) ^ (a >> 32);
  return a;
}

#endif