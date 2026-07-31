// tiles.go
package main

import (
	"fmt"
	"math/bits"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/holiman/uint256"
)

// wideNum is the result-accumulator type: a 256-bit unsigned integer.
// Counts for maxPips=8 reach roughly 10^25 (~84 bits) and can plausibly go
// higher at mid-array indices, comfortably exceeding uint64's ~1.8*10^19
// ceiling. uint256.Int is a plain [4]uint64 under the hood (not a
// pointer/slice type like math/big.Int), which matters here: these values
// are returned and copied by ordinary Go value semantics all through the
// recursion (try()'s return value, sub[k], local[k], task structs, ...), and
// a slice-backed bignum's naive struct copies can alias a shared backing
// array across what should be independent values - a real correctness trap
// in code shaped like this one. A fixed array has no such hazard: copying it
// is always a true, independent copy. The wrappers below just adapt
// uint256.Int's pointer-receiver, mutate-in-place API to this codebase's
// value-in-value-out style.
type wideNum = uint256.Int

func wideFromU64(v uint64) wideNum {
	var r wideNum
	r.SetUint64(v)
	return r
}

func wideAdd(a, b wideNum) wideNum {
	var r wideNum
	r.Add(&a, &b)
	return r
}

// mulSmall multiplies a (256-bit) by m (a plain uint64 scalar - branchMult,
// mult and symmetry are all small combinatorial multipliers, never
// themselves 256-bit, so a full wideNum*wideNum multiply is never needed).
func mulSmall(a wideNum, m uint64) wideNum {
	var mm, r wideNum
	mm.SetUint64(m)
	r.Mul(&a, &mm)
	return r
}

// maxNumValues bounds the compiled-in size of the hot-path arrays (pused,
// rowSums, myres/local). It must be at least maxPips+1. 11 covers maxPips up
// to 10; bump it (and rebuild) to go higher - see setup's panic message. It
// can't go past 11 as-is regardless: canonicalKey packs the edge set into a
// uint64 mask, needing C(numValues,2) bits, and C(12,2)=66 already overflows
// 64 - see setup's check, which turns that into a panic rather than letting
// it silently collide different states onto the same memo key.
const maxNumValues = 11
const maxDominoes = maxNumValues * (maxNumValues + 1) / 2 // 45

// maxPips is the highest pip value on a tile (so values run 0..maxPips).
// Must be >= 3 (odd values are supported - see setup - but for odd maxPips no
// single chain can use every tile; see setup's comment). Set this (or pass
// -pips on the command line, see main.go) before calling mainLoop;
// setup(maxPips) - called by mainLoop itself - derives everything else and is
// a no-op if already configured for the requested value, so repeated
// mainLoop() calls (e.g. from a benchmark loop) only pay the cache-population
// cost once.
var maxPips = 6

// Derived from maxPips by setup(); see setup for how each is computed.
var numValues, numEdges, totalDominoes, cDim int
var gogo int // level to parallelize on

var configuredPips = -1

// gogoOverride lets a caller (see main.go's -gogo flag) force the
// parallelization depth instead of the numValues-1 default, since the right
// tradeoff between goroutine count (parallelism grain, peak memory per
// goroutine) and per-goroutine memoization payoff (bigger subtrees revisit
// states more) shifts with maxPips. -1 means "use the default".
var gogoOverride = -1

var mu sync.Mutex
var res [maxDominoes]wideNum
var used, used4 [maxNumValues][maxNumValues]uint8
var wg sync.WaitGroup

// goroutinesSpawned/Completed are cheap atomic counters a caller (see
// main.go) can poll from a separate goroutine to report progress, without
// mainLoop itself doing any I/O - which keeps `go test -bench=Main` a clean
// measurement of the search alone. Despite the name, with the worker pool
// below they now count queued/finished tasks rather than actual OS
// goroutines.
var goroutinesSpawned int64
var goroutinesCompleted int64

// task is one unit of work handed to the worker pool: continue try() from
// (n, lastRight) with this (copied, independent) board state.
type task struct {
	n, lastRight, symmetry int
	pused                  [maxNumValues][maxNumValues]uint8
	rowSums                [maxNumValues]uint8
}

// taskQueue is an unbounded, mutex-protected LIFO queue feeding the worker
// pool. It exists because a plain buffered channel can't safely stand in for
// gotry()'s old "just spawn a goroutine" behavior: gotry() is called from
// *inside* a worker's own try() recursion (once n reaches gogo), so if
// pushing new work could ever block, every worker could end up blocked
// trying to push while none are left to pop - deadlock. A slice append under
// a mutex never blocks the producer, so that can't happen; the price is that
// backlog sits as ~100-byte task structs instead of active goroutines, which
// is what actually fixes the memory blowup (a queued task is orders of
// magnitude cheaper than a goroutine stack).
type taskQueue struct {
	mu    sync.Mutex
	cond  *sync.Cond
	items []task
}

func newTaskQueue() *taskQueue {
	q := &taskQueue{}
	q.cond = sync.NewCond(&q.mu)
	return q
}

func (q *taskQueue) push(t task) {
	q.mu.Lock()
	q.items = append(q.items, t)
	q.mu.Unlock()
	q.cond.Signal()
}

// pop blocks until an item is available. Workers run forever, so this never
// needs a "queue closed" case - between mainLoop() calls (e.g. in a
// benchmark loop) workers simply sit idle here.
func (q *taskQueue) pop() task {
	q.mu.Lock()
	defer q.mu.Unlock()
	for len(q.items) == 0 {
		q.cond.Wait()
	}
	last := len(q.items) - 1
	t := q.items[last]
	q.items = q.items[:last]
	return t
}

var taskQ = newTaskQueue()
var workerPoolOnce sync.Once

// ensureWorkers starts a fixed pool of GOMAXPROCS persistent workers on
// first use. Bounding the pool size, not just gating entry to the expensive
// part, is what keeps active goroutine count (and thus total stack memory)
// flat regardless of how many tasks the search tree generates.
func ensureWorkers() {
	workerPoolOnce.Do(func() {
		for range runtime.GOMAXPROCS(0) {
			go worker()
		}
	})
}

func worker() {
	for {
		t := taskQ.pop()
		runTry(t.n, t.lastRight, t.symmetry, t.pused, t.rowSums)
	}
}

// sharedMemo replaces the old per-goroutine memo maps: with one map per
// worker, a state recurring across *different* top-level tasks (not just
// within one task's own subtree) was recomputed once per task instead of
// once total - real duplicated work, confirmed by shallower gogo (bigger,
// better-deduplicated per-task subtrees, less cross-task loss) outrunning
// deeper gogo despite using far less parallelism. Sharing one memo across
// every worker fixes that at the cost of needing synchronization; sharding
// by key across many independently-locked buckets keeps that synchronization
// cheap by spreading contention out, rather than serializing every worker
// through one global lock.
const memoShardCount = 512

type memoShard struct {
	mu    sync.Mutex
	m     map[uint64][maxDominoes]wideNum
	churn int // inserts-while-at-capacity since the last rebuild; see memoPut
}

var memoShards [memoShardCount]memoShard

// memoMaxEntries caps the shared memo's TOTAL size across all shards; 0
// means unbounded (the original behavior, still the default - every pips
// value validated so far, 3 through 9, comfortably fits in real RAM
// unbounded). It exists because the state space at numValues=11 (maxPips=10)
// turned out to be flatly enormous - a real run there hit ~60GB with only 1
// of 14084 top-level tasks even finished, and was killed (almost certainly
// by the OOM killer; the box has no swap) before completing even one. No
// amount of RAM headroom fixes an unbounded map growing that fast; capping
// it and evicting when full trades some recomputation (an evicted-then-
// needed-again state just gets recomputed, which is correct, only slower)
// for a hard ceiling that turns "OOM the whole box" into "predictably use
// this much memory, however long that takes." Set via -memomb (see
// main.go), which converts a budget in MB to an entry count using the
// actual per-entry size for the current numValues.
var memoMaxEntries int64 = 0

// memoShardCap returns the per-shard entry cap (0 = unbounded), dividing
// memoMaxEntries evenly across the shards.
func memoShardCap() int {
	if memoMaxEntries <= 0 {
		return 0
	}
	c := int(memoMaxEntries / memoShardCount)
	if c < 1 {
		c = 1
	}
	return c
}

// resetMemo (re)allocates every shard. Called once per mainLoop() run so
// repeated calls (e.g. from a benchmark loop) each measure a cold cache
// instead of getting artificially faster from a previous run's leftovers.
func resetMemo() {
	for i := range memoShards {
		memoShards[i].mu.Lock()
		memoShards[i].m = make(map[uint64][maxDominoes]wideNum)
		memoShards[i].churn = 0
		memoShards[i].mu.Unlock()
	}
}

func memoGet(key uint64) ([maxDominoes]wideNum, bool) {
	s := &memoShards[key%memoShardCount]
	s.mu.Lock()
	v, ok := s.m[key]
	s.mu.Unlock()
	return v, ok
}

// memoPut stores v, evicting one entry first if the shard is already at
// capacity and key isn't already present. Go's map iteration order is
// randomized per-iteration, so "delete whatever the range loop visits
// first" is a cheap, simple random-eviction policy - not as effective as a
// true LRU at keeping "hot" entries, but needs no extra bookkeeping, and a
// worse eviction choice only costs a future recomputation, never
// correctness.
//
// Eviction alone isn't sufficient to actually bound memory, though: Go's map
// never shrinks its bucket storage on delete(), so under sustained
// evict-then-insert churn the underlying allocation keeps growing to match
// the *cumulative* number of distinct keys ever inserted, not the current
// live (capped) entry count - confirmed empirically, a 20GB cap actually
// used ~42GB before this fix. Periodically copying the live entries into a
// freshly allocated map reclaims that bloat. Rebuilding every cap/4 evictions
// bounds the worst-case overshoot to roughly 1.25x the cap instead of
// growing without limit - except for very small caps, where cap/4 rounds
// down toward 0 and would rebuild on nearly every single eviction, which is
// pathologically expensive at the millions-of-calls-per-second this runs at
// (confirmed empirically: a 1MB budget, cap=1/shard, hung for minutes on a
// case that finishes instantly unbounded). rebuildEvery has a floor for
// exactly that reason - a small cap already wastes little memory even with a
// generous rebuild interval, since the absolute entry count is tiny.
func memoPut(key uint64, v [maxDominoes]wideNum) {
	s := &memoShards[key%memoShardCount]
	cap := memoShardCap()
	s.mu.Lock()
	if cap > 0 && len(s.m) >= cap {
		if _, exists := s.m[key]; !exists {
			for k := range s.m {
				delete(s.m, k)
				break
			}
			s.churn++
			rebuildEvery := cap / 4
			if rebuildEvery < 256 {
				rebuildEvery = 256
			}
			if s.churn >= rebuildEvery {
				fresh := make(map[uint64][maxDominoes]wideNum, len(s.m))
				for k2, v2 := range s.m {
					fresh[k2] = v2
				}
				s.m = fresh
				s.churn = 0
			}
		}
	}
	s.m[key] = v
	s.mu.Unlock()
}

func gotry(n, lastRight, symmetry int, pused *[maxNumValues][maxNumValues]uint8, rowSums *[maxNumValues]uint8) {
	ensureWorkers()
	atomic.AddInt64(&goroutinesSpawned, 1)
	wg.Add(1)
	taskQ.push(task{n, lastRight, symmetry, *pused, *rowSums})
}

// runTry drives one worker's share of the search: try() is a pure function
// of the search state, so its result can be memoized (now in the shared,
// cross-worker memoShards - see try) and only needs to be scaled by symmetry
// and folded into the shared res once, here, at the end.
func runTry(n, lastRight, symmetry int, pused [maxNumValues][maxNumValues]uint8, rowSums [maxNumValues]uint8) {
	result := try(n, lastRight, symmetry, &pused, &rowSums, 1)
	mu.Lock()
	for i := 0; i < totalDominoes; i++ {
		// symmetry is always a product of small positive combinatorial
		// factors (numValues-1, numValues-2, ... and branchMult multipliers,
		// see mainLoop/try) - never negative, nowhere near uint64 range.
		res[i] = wideAdd(res[i], mulSmall(result[i], uint64(symmetry))) // #nosec G115
	}
	mu.Unlock()
	atomic.AddInt64(&goroutinesCompleted, 1)
	wg.Done()
}

// edgeBit[i][j] gives each of the numEdges unordered vertex pairs a unique
// bit position, used to pack an edge set into a compact memo/state key.
var edgeBit [maxNumValues][maxNumValues]int

// setup derives every size from maxPips and rebuilds edgeBit. It is
// idempotent for a given pips value, so calling it (as mainLoop does) on
// every call is cheap after the first.
//
// This used to also precompute and hold two huge lookup tables (an
// elementary-symmetric-polynomial cache and a same-rowSum-value bitmask
// table) sized by every reachable rowSums tuple - correct, but for
// maxPips=10 that alone needs ~30GB, most of it for tuples the (heavily
// deduplicated) search never actually visits. Both tables are now computed
// on demand instead - see addDoubles and try's valueGroup - trading a small
// per-call CPU cost for eliminating that memory (and its build time: 44.9s
// measured for maxPips=10) entirely. This also drops the even/odd maxSum0
// bound derivation that used to size the tables: the on-demand DP is correct
// for any nonnegative rowSums values, so there's no encoding to alias if
// that bound were ever wrong.
func setup(pips int) {
	if pips == configuredPips {
		return
	}
	if pips < 3 {
		panic(fmt.Sprintf("maxPips must be >= 3, got %d", pips))
	}
	nv := pips + 1
	if nv > maxNumValues {
		panic(fmt.Sprintf("maxPips=%d needs numValues=%d, but maxNumValues is compiled in as %d - bump the constant and rebuild", pips, nv, maxNumValues))
	}
	if nv*(nv-1)/2 > 64 {
		panic(fmt.Sprintf("maxPips=%d needs numValues=%d, whose edge count C(%d,2)=%d exceeds the 64 bits canonicalKey's mask can hold - widen that mask type before going this high", pips, nv, nv, nv*(nv-1)/2))
	}

	numValues = nv
	numEdges = numValues * (numValues - 1) / 2
	totalDominoes = numValues * (numValues + 1) / 2
	cDim = numValues + 1

	bit := 0
	for i := 0; i < numValues; i++ {
		for j := i + 1; j < numValues; j++ {
			edgeBit[i][j] = bit
			edgeBit[j][i] = bit
			bit++
		}
	}

	configuredPips = pips
}

// applyGogo sets gogo from gogoOverride (if set) or the numValues-1 default.
// Unlike the rest of setup, it always runs - even when setup() short-circuits
// because pips hasn't changed - so gogoOverride can be tuned between
// mainLoop() calls without forcing a cache rebuild.
func applyGogo() {
	if gogoOverride >= 0 {
		gogo = gogoOverride
	} else {
		gogo = numValues - 1
	}
}

// addDoubles adds cache[c] = e_c(sums[0..numValues-1]), the c-th elementary
// symmetric polynomial, to target[n+c] for c=0..numValues, each scaled by
// mult. e_0..e_numValues are computed directly via the standard O(numValues^2)
// DP (e_k(x_1..x_n) = e_k(x_1..x_{n-1}) + x_n*e_{k-1}(x_1..x_{n-1}): fold in
// one variable at a time, updating e[k] from high k to low k so a variable's
// own update isn't reused within the same step) instead of a precomputed
// table lookup - see setup's comment for why.
func addDoubles(n int, target *[maxDominoes]wideNum, sums *[maxNumValues]uint8, mult uint64) {
	var e [maxNumValues + 1]uint32
	e[0] = 1
	for v := 0; v < numValues; v++ {
		x := uint32(sums[v])
		for k := v + 1; k >= 1; k-- {
			e[k] += x * e[k-1]
		}
	}
	for c := 0; c <= numValues; c++ {
		target[n] = wideAdd(target[n], mulSmall(wideFromU64(uint64(e[c])), mult))
		n++
	}
}

// canonicalKey computes a canonical form of the (remaining-edges graph,
// lastRight) state under relabeling of the non-lastRight vertices, so that
// isomorphic states - not just literal duplicates - share a memo entry. This
// is a much larger symmetry group than stateKey's plain edge-bitmask
// exploits (only exact repeats): any permutation of the other numValues-1
// vertices that maps the graph to itself gives an identical future subtree,
// by the same argument that justifies sibling-twin collapsing in try()
// (addDoubles's cache lookups are permutation-invariant elementary symmetric
// polynomials), just extended from swapping one pair to relabeling
// everything.
//
// Vertices are colored by (rowSums value, adjacency to lastRight), then
// refined a few rounds by each vertex's multiset of neighbor colors
// (Weisfeiler-Leman style) to split apart vertices that can't possibly be
// interchanged. This coloring is a graph invariant - isomorphic states get
// matching color sequences - so grouping by color and searching only within
// each color class for the minimal encoding finds a true canonical form; a
// coarser-than-necessary grouping (e.g. from hash collisions) only costs
// extra permutations tried, it never breaks correctness, since the search
// still covers every needed rearrangement.
func canonicalKey(pused *[maxNumValues][maxNumValues]uint8, lastRight int, rowSums *[maxNumValues]uint8) uint64 {
	var others [maxNumValues]int
	no := 0
	for v := 0; v < numValues; v++ {
		if v != lastRight {
			others[no] = v
			no++
		}
	}

	var color [maxNumValues]int
	for i := 0; i < no; i++ {
		v := others[i]
		adj := 0
		if pused[lastRight][v] != 0 {
			adj = 1
		}
		color[v] = int(rowSums[v])*2 + adj
	}

	var neigh [maxNumValues]int
	for round := 0; round < 3; round++ {
		var next [maxNumValues]int
		for i := 0; i < no; i++ {
			v := others[i]
			nn := 0
			for j := 0; j < no; j++ {
				u := others[j]
				if u != v && pused[v][u] != 0 {
					neigh[nn] = color[u]
					nn++
				}
			}
			sort.Ints(neigh[:nn])
			h := color[v]
			for k := 0; k < nn; k++ {
				h = h*1000003 + neigh[k] + 1
			}
			next[v] = h
		}
		for i := 0; i < no; i++ {
			color[others[i]] = next[others[i]]
		}
	}

	sortedOthers := make([]int, no)
	copy(sortedOthers, others[:no])
	sort.Slice(sortedOthers, func(a, b int) bool {
		ca, cb := color[sortedOthers[a]], color[sortedOthers[b]]
		if ca != cb {
			return ca < cb
		}
		return sortedOthers[a] < sortedOthers[b]
	})

	var classes [][]int
	for i := 0; i < no; {
		j := i + 1
		for j < no && color[sortedOthers[j]] == color[sortedOthers[i]] {
			j++
		}
		classes = append(classes, sortedOthers[i:j])
		i = j
	}

	bestMask := ^uint64(0)
	var positions [maxNumValues]int
	var full [maxNumValues]int
	full[0] = lastRight

	evaluate := func() {
		for i := 0; i < no; i++ {
			full[i+1] = positions[i]
		}
		var mask uint64
		for a := 0; a < numValues; a++ {
			for b := a + 1; b < numValues; b++ {
				if pused[full[a]][full[b]] != 0 {
					mask |= 1 << uint(edgeBit[a][b])
				}
			}
		}
		if mask < bestMask {
			bestMask = mask
		}
	}

	var permuteClass func(classIdx, offset int)
	permuteClass = func(classIdx, offset int) {
		if classIdx == len(classes) {
			evaluate()
			return
		}
		cls := classes[classIdx]
		var perm func(k int)
		buf := make([]int, len(cls))
		copy(buf, cls)
		perm = func(k int) {
			if k == len(buf) {
				for i, v := range buf {
					positions[offset+i] = v
				}
				permuteClass(classIdx+1, offset+len(buf))
				return
			}
			for i := k; i < len(buf); i++ {
				buf[k], buf[i] = buf[i], buf[k]
				perm(k + 1)
				buf[k], buf[i] = buf[i], buf[k]
			}
		}
		perm(0)
	}
	permuteClass(0, 0)

	return bestMask
}

// try explores every unused edge out of lastRight for position n onward and
// returns the total contribution to myres[n..totalDominoes-1] as if this
// exact subtree were reached exactly once. Two distinct kinds of duplicate
// work are avoided:
//
//   - Sibling twins: candidates with the same row sum and identical adjacency
//     to every other vertex give identical subtrees - swapping which of them
//     gets used next leaves the multiset of row sums, and hence every future
//     cache lookup (which depends only on that multiset), unchanged. They are
//     found via valueGroup (computed fresh each call, same idea as the old
//     precomputed sameValMask table - see addDoubles/setup for why it isn't
//     precomputed anymore) narrowed down with a single-XOR adjacency check,
//     and folded into one representative scaled by branchMult.
//   - Repeated states: past the goroutine-spawn depth (gogo), the exact same
//     (used-edge-set, lastRight) state recurs enormously often via different
//     edge orderings (empirically over 99% of calls past that depth are
//     revisits, for maxPips=6), so results are memoized per goroutine keyed
//     by stateKey.
//
// mult is the cumulative branchMult collected from ancestor levels within
// this same goroutine (1 at a goroutine's own entry point). The synchronous
// return-value chain (local[k] += sub[k]*branchMult) already folds ancestor
// branchMult into results that bubble up normally, but a gotry() spawn at
// n==gogo escapes that chain - its result is scaled by symmetry alone in
// runTry, so symmetry must be pre-multiplied by mult*branchMult at the call
// site to still account for duplicates collapsed above it. mult is never
// used past n==gogo: memoization only starts at n>gogo, strictly after the
// single point gotry can be called from, so the two never interact.
func try(n, lastRight, symmetry int, pused *[maxNumValues][maxNumValues]uint8, rowSums *[maxNumValues]uint8, mult uint64) [maxDominoes]wideNum {
	memoize := n > gogo
	var key uint64
	if memoize {
		key = canonicalKey(pused, lastRight, rowSums)
		if cached, ok := memoGet(key); ok {
			return cached
		}
	}

	var rowMask [maxNumValues]uint16
	var valueGroup [maxNumValues]uint16
	var candMask uint16
	for i := 0; i < numValues; i++ {
		var m uint16
		row := &pused[i]
		for k := 0; k < numValues; k++ {
			m |= uint16(row[k]) << k
		}
		rowMask[i] = m
		if pused[lastRight][i] == 0 {
			candMask |= 1 << i
		}
	}
	for p := 0; p < numValues; p++ {
		var m uint16
		for q := 0; q < numValues; q++ {
			if rowSums[p] == rowSums[q] {
				m |= 1 << q
			}
		}
		valueGroup[p] = m
	}

	var local [maxDominoes]wideNum
	var doneMask uint16
	for cm := candMask; cm != 0; cm &= cm - 1 {
		i := bits.TrailingZeros16(cm)
		if doneMask&(1<<i) != 0 {
			continue
		}
		branchMult := uint64(1)
		group := valueGroup[i] & candMask &^ doneMask &^ (1 << i)
		for g := group; g != 0; g &= g - 1 {
			j := bits.TrailingZeros16(g)
			if (rowMask[i]^rowMask[j])&^((1<<i)|(1<<j)) == 0 {
				doneMask |= 1 << j
				branchMult++
			}
		}
		pused[lastRight][i] = 1
		pused[i][lastRight] = 1
		rowSums[i]++
		addDoubles(n, &local, rowSums, branchMult)
		if n+1 < numEdges {
			if n == gogo {
				// mult*branchMult is a product of at most gogo (< numEdges)
				// factors each <= numValues <= maxNumValues (9), so it's
				// nowhere near overflowing int here; would need revisiting
				// only alongside bumping maxNumValues itself (see its
				// comment) to something drastically larger.
				gotry(n+1, i, symmetry*int(mult*branchMult), pused, rowSums) // #nosec G115
			} else {
				sub := try(n+1, i, symmetry, pused, rowSums, mult*branchMult)
				for k := 0; k < totalDominoes; k++ {
					local[k] = wideAdd(local[k], mulSmall(sub[k], branchMult))
				}
			}
		}
		pused[lastRight][i] = 0
		pused[i][lastRight] = 0
		rowSums[i]--
	}

	if memoize {
		// Two workers can race to compute the same state and both miss the
		// cache before either stores it; that's wasted duplicate work, not a
		// correctness issue (both compute the same value), and adding a
		// claim-the-computation lock to close that gap isn't worth it against
		// how sparse genuine collisions are across this large a key space.
		memoPut(key, local)
	}
	return local
}

func play(n int, tile [2]int, symmetry int, pused *[maxNumValues][maxNumValues]uint8, rowSums *[maxNumValues]uint8) {
	pused[tile[0]][tile[1]] = 1
	pused[tile[1]][tile[0]] = 1
	rowSums[tile[1]]++
	var myres [maxDominoes]wideNum
	addDoubles(n, &myres, rowSums, 1)
	mu.Lock()
	for i := 0; i < totalDominoes; i++ {
		res[i] = wideAdd(res[i], mulSmall(myres[i], uint64(symmetry))) // #nosec G115 -- see runTry
	}
	mu.Unlock()
}

var initialSym uint64

func mainLoop() {
	setup(maxPips)
	applyGogo()
	resetMemo()

	var rowSums [maxNumValues]uint8
	for i := 0; i < numValues; i++ { // block doubles
		used[i][i] = 1
	}
	rowSums[0] = 1

	res = [maxDominoes]wideNum{}
	res[0] = wideFromU64(1)
	initialSym = uint64(numValues) // #nosec G115 -- numValues <= maxNumValues (9)

	// first tile 0|1, x(numValues-1): value 1 could be any of the
	// numValues-1 nonzero values, all equivalent by relabeling.
	symmetry := numValues - 1
	play(0, [2]int{0, 1}, symmetry, &used, &rowSums)
	// second tile 1|2, x(numValues-2): the new value is any of the
	// remaining numValues-2 choices.
	symmetry *= numValues - 2
	play(1, [2]int{1, 2}, symmetry, &used, &rowSums)
	used4 = used // copy for the other set of workers
	rowSums4 := rowSums
	// 2|0 - 1st option on 3rd step, x1: closing back to 0.
	play(2, [2]int{2, 0}, symmetry, &used, &rowSums)
	gotry(3, 0, symmetry, &used, &rowSums) // do not wait yet
	// 2|3 - 2nd option on 3rd step, x(numValues-3): a genuinely new value.
	symmetry *= numValues - 3
	play(2, [2]int{2, 3}, symmetry, &used4, &rowSums4)
	gotry(3, 3, symmetry, &used4, &rowSums4)
	wg.Wait() // wait

}
