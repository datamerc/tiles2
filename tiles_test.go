package main

import "testing"

func BenchmarkMain(b *testing.B) {
	setup(maxPips) // pay cache-population cost once, outside the timed loop

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		mainLoop()
	}
}
