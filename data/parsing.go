package data

import "strconv"

type Range struct{
	Min Float64
	Max Float64
}

var Range0to1 = Range{Float64(0), Float64(1)}
var RangeN1to1 = Range{Float64(-1), Float64(1)}

func NewRange(min, max float64) Range{
	return Range{Float64(min), Float64(max)}
}

type Float64 float64

func ToFloat64(s string, defValue float64) Float64{
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {return Float64(defValue)}
	return Float64(f)
}

func (f Float64) Normalise(old, new Range) Float64{
	return ((f-old.Min)/(old.Max-old.Min)) * (new.Max-new.Min) + new.Min
}

func (f Float64)Clamp(rng Range) Float64{
	if f < rng.Min{
		return rng.Min
	} else if f > rng.Max{
		return rng.Max
	}
	return f
}


func (f Float64)Parse()float64{
	return float64(f)
}


// Classification

// The functions below assume that ClassificationLabels is not dodgy (labels from 0-n)

type ClassificationLabels map[string]int
func NewClassLabels(labels []string)ClassificationLabels{
	cls := make(ClassificationLabels, 0)
	for i, v := range labels{
		cls[v] = i
	}
	return cls
}

func ClassifyBinary(s string, labels ClassificationLabels, defClass string) float64{
	if v, ok := labels[s]; ok{
		return float64(v)
	}
	if v, ok := labels[defClass]; ok{
		return float64(v)
	}
	return 0
}

func ClassifyCategorical(s string, labels ClassificationLabels, defClass string) []float64{
	V := 0
	if v, ok := labels[s]; ok{
		V = v
	} else if v, ok := labels[defClass]; ok{
		V = v
	}
	parts := make([]float64, len(labels))
	for i := range parts{
		if i == V{
			parts[i] = 1
		} else{
			parts[i] = 0
		}
	}
	return parts
}

func (r Row)CopyFrom(i int, fs []float64){
	for j, f := range fs {
		r[i+j] = f
	}
}

func AppendAll(cols... []float64)[]float64{
	l := 0
	for i := range cols{
		l += len(cols[i])
	}
	out := make([]float64, l)
	o := 0
	for i := range cols{
		for _, v := range cols[i]{
			out[o] = v
			o++
		}
	}
	return out
}