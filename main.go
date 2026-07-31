package main

import (
	"flag"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync/atomic"
	"syscall"
	"time"
)

// HoldingRusage fetches current process resource metrics.
func HoldingRusage() syscall.Rusage {
	var usage syscall.Rusage
	// Diagnostic-only (resource reporting below); RUSAGE_SELF on the
	// current process has no realistic failure mode worth handling.
	_ = syscall.Getrusage(syscall.RUSAGE_SELF, &usage)
	return usage
}

// maxrssDeltaMB converts a Maxrss delta to MB. Rusage.Maxrss's unit differs
// by platform - bytes on Darwin, kilobytes on Linux/most other POSIX systems
// - dividing by the wrong factor silently under-reports Linux figures by
// 1024x, which matters a lot when the whole point of a run is deciding
// whether a box's RAM is enough.
func maxrssDeltaMB(deltaRaw int64) int64 {
	if runtime.GOOS == "darwin" {
		return deltaRaw / 1024 / 1024
	}
	return deltaRaw / 1024
}

// bytesPerMemoEntry estimates one shared-memo entry's footprint: the value
// ([maxDominoes]wideNum, each wideNum a 256-bit/32-byte uint256.Int), the
// uint64 key, and Go's per-entry map bucket overhead (conservatively
// estimated - exact figure depends on Go version/bucket occupancy, but
// erring high just means memomb caps slightly smaller than requested,
// never the dangerous direction).
const bytesPerMemoEntry = maxDominoes*32 + 8 + 100

func main() {
	pips := flag.Int("pips", 6, "maximum pip value (>= 3; odd values have no full-length single chain)")
	gogoFlag := flag.Int("gogo", -1, "parallelization depth override (-1 = auto, numValues-1)")
	memoMB := flag.Int64("memomb", 0, "cap the shared memo to roughly this many MB total (0 = unbounded, the default and fine through at least maxPips=9; maxPips=10's state space is large enough to OOM even a 62GB box unbounded - see memoMaxEntries)")
	memLimitMB := flag.Int64("memlimitmb", 0, "set a Go soft heap limit (runtime/debug.SetMemoryLimit) in MB, making GC collect aggressively as usage approaches it instead of letting garbage - evicted memo values, arithmetic temporaries - pile up before a collection cycle (0 = don't set one, Go's default GC pacing)")
	flag.Parse()
	maxPips = *pips
	gogoOverride = *gogoFlag
	if *memLimitMB > 0 {
		debug.SetMemoryLimit(*memLimitMB * 1024 * 1024)
	}
	if *memoMB > 0 {
		memoMaxEntries = *memoMB * 1024 * 1024 / bytesPerMemoEntry
	}

	// 1. Capture resource snapshot BEFORE the procedure
	startRusage := HoldingRusage()
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	startTime := time.Now()

	// Progress is reported from a goroutine outside mainLoop, via the
	// atomic counters it updates, so mainLoop itself stays free of any I/O
	// and can be benchmarked cleanly (see BenchmarkMain).
	progressStop := make(chan struct{})
	progressDone := make(chan struct{})
	go func() {
		defer close(progressDone)
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		var lastSpawned, lastCompleted int64
		for {
			select {
			case <-ticker.C:
				spawned := atomic.LoadInt64(&goroutinesSpawned)
				completed := atomic.LoadInt64(&goroutinesCompleted)
				if spawned != lastSpawned || completed != lastCompleted {
					fmt.Printf("progress: %d/%d goroutines complete\n", completed, spawned)
					lastSpawned, lastCompleted = spawned, completed
				}
			case <-progressStop:
				return
			}
		}
	}()

	// Main loop
	mainLoop()

	close(progressStop)
	<-progressDone
	// --- YOUR PROCEDURE END ---

	// 2. Capture resource snapshot AFTER the procedure
	endRusage := HoldingRusage()
	endTime := time.Since(startTime)

	// 3. Calculate differences
	// User CPU time (seconds + microseconds)
	startUserCPU := time.Duration(startRusage.Utime.Sec)*time.Second + time.Duration(startRusage.Utime.Usec)*time.Microsecond
	endUserCPU := time.Duration(endRusage.Utime.Sec)*time.Second + time.Duration(endRusage.Utime.Usec)*time.Microsecond
	userCPUDiff := endUserCPU - startUserCPU

	// System/Kernel CPU time
	startSysCPU := time.Duration(startRusage.Stime.Sec)*time.Second + time.Duration(startRusage.Stime.Usec)*time.Microsecond
	endSysCPU := time.Duration(endRusage.Stime.Sec)*time.Second + time.Duration(endRusage.Stime.Usec)*time.Microsecond
	sysCPUDiff := endSysCPU - startSysCPU

	// Memory footprint difference (Max Resident Set Size).
	// Note: MaxRSS tracks peak memory, so it shows the high-water mark during execution.
	memDiffMB := maxrssDeltaMB(endRusage.Maxrss - startRusage.Maxrss)
	// 4. Calculate CPU / Wall-clock ratio
	var cpuRatio float64
	if endTime > 0 {
		cpuRatio = float64(userCPUDiff+sysCPUDiff) / float64(endTime)
	}
	runtime.GC()
	runtime.ReadMemStats(&after)

	fmt.Println("HeapAlloc delta:", after.HeapAlloc-before.HeapAlloc)
	fmt.Println("TotalAlloc delta:", after.TotalAlloc-before.TotalAlloc)
	fmt.Println("Mallocs delta:", after.Mallocs-before.Mallocs)
	fmt.Println("Frees delta:", after.Frees-before.Frees)

	// 4. Report results
	fmt.Println("\n--- Procedure Resource Diff ---")
	fmt.Printf("Wall-clock time elapsed: %v\n", endTime)
	fmt.Printf("User CPU time used:      %v\n", userCPUDiff)
	fmt.Printf("System CPU time used:    %v\n", sysCPUDiff)
	fmt.Printf("Total CPU time used:     %v\n", userCPUDiff+sysCPUDiff)
	fmt.Printf("CPU / Wall-clock ratio:   %.2f (%.0f%%)\n", cpuRatio, cpuRatio*100)
	fmt.Printf("Peak RAM change (MaxRSS): %d MB\n", memDiffMB)
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	fmt.Println("HeapAlloc:", m.HeapAlloc)
	fmt.Println("HeapSys:", m.HeapSys)
	fmt.Println("NumGC:", m.NumGC)
	fmt.Println("goroutines:", runtime.NumGoroutine())

	for i := 0; i < totalDominoes; i++ {
		v := mulSmall(res[i], initialSym)
		fmt.Println(i+1, v.Dec())
	}
}
