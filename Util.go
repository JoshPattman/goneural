package goneural

import (
	"math/rand"
	"time"
)

// makeRandomSlice : Generated a slice of floats between minVal and maxVal
func makeRandomSlice(length int, minVal, maxVal float64)[]float64{
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	s := make([]float64, length)
	for i := range s{
		s[i] = r.Float64() * (maxVal - minVal) + minVal
	}
	return s
}
