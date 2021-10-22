package goneural

type Matrix struct{
	Shape []int
	Values []float64
}

func New1DMatrix(vals []float64) *Matrix{
	return &Matrix{
		Shape: []int{len(vals)},
		Values: vals,
	}
}
func New2DMatrix(vals [][]float64) *Matrix{
	sqVals := make([]float64, len(vals)* len(vals[0]))
	i := 0
	for x := range vals{
		for y := range vals[x]{
			sqVals[i] = vals[x][y]
			i++
		}
	}
	return &Matrix{
		Shape: []int{len(vals), len(vals[0])},
		Values: sqVals,
	}
}

func NewZerosMatrix(shape []int)*Matrix{
	l := 1
	for _, l1 := range shape{
		l *= l1
	}
	return &Matrix{
		Shape: shape,
		Values: make([]float64, l),
	}
}

func (m *Matrix) GetValue1D(i int)float64{
	return m.Values[i]
}

func (m *Matrix) SetValue1D(i int, v float64){
	m.Values[i] = v
}

func (m *Matrix)GetValue2D(x, y int) float64{
	i := x*m.Shape[1] + y
	return m.Values[i]
}

func (m *Matrix)SetValue2D(x, y int, v float64){
	i := x*m.Shape[1] + y
	m.Values[i] = v
}

func (m *Matrix) Get1DLength()int{
	return m.Shape[0]
}

func (m *Matrix)Mul( f float64) *Matrix{
	for i := range m.Values{
		m.Values[i] = m.Values[i] * f
	}
	return m
}

func (m *Matrix) CopyTo(m2 *Matrix){
	for v := range m.Values{
		m2.Values[v] = m.Values[v]
	}
}