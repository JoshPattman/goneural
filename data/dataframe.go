package data

import (
	"encoding/csv"
	"github.com/JoshPattman/goneural"
	"log"
	"os"
)

type RawTable struct{
	ColumnNames []string
	Rows []RawRow
}

type Table struct{
	ColumnNames []string
	Rows []Row
}



func (rt *RawTable) Convert(function ParseFunction) *Table{
	t := &Table{
		rt.ColumnNames,
		make([]Row, len(rt.Rows)),
	}
	for i := range rt.Rows{
		t.Rows[i] = rt.Rows[i].Convert(function)
	}
	return t
}

func (t *Table)Convert(x []int, y []int) ([]*goneural.Matrix, []*goneural.Matrix){
	dataX := make([]*goneural.Matrix, len(t.Rows))
	dataY := make([]*goneural.Matrix, len(t.Rows))
	for r := range dataX{
		X := make([]float64, 0)
		Y := make([]float64, 0)
		for c := range t.Rows[r]{
			if InXs(c, x){
				X = append(X, t.Rows[r][c])
			}
			if InXs(c, y){
				Y = append(Y, t.Rows[r][c])
			}
		}
		dataX[r] = goneural.New1DMatrix(X)
		dataY[r] = goneural.New1DMatrix(Y)
	}
	return dataX, dataY
}

func InXs(i int, xs []int) bool{
	for ix := range xs{
		if xs[ix] == i{
			return true
		}
	}
	return false
}

func (rt *RawTable)GetClassesFromColumnIndex(columnIndex int) ClassificationLabels{
	classes := make(map[string]int, 0)
	for i := range rt.Rows{
		classes[rt.Rows[i][columnIndex]] = 0
	}
	i := 0
	for k, _ := range classes{
		classes[k] = i
		i++
	}
	return ClassificationLabels(classes)
}
func (rt *RawTable)GetScalesFromColumnIndex(columnIndex int) Range{
	rng := NewRange(10000000,-10000000)
	for i := range rt.Rows{
		n := ToFloat64(rt.Rows[i][columnIndex], -123456789)
		if n != -123456789{
			if n < rng.Min{rng.Min = n}
			if n > rng.Max{rng.Max = n}
		}
	}
	return rng
}

func RawTableFromCSV(filePath string) RawTable {
	f, err := os.Open(filePath)
	if err != nil {
		log.Fatal("Unable to read input file " + filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		log.Fatal("Unable to parse file as CSV for " + filePath, err)
	}

	t := RawTable{
		records[0],
		make([]RawRow, len(records)-1),
	}

	for i := range t.Rows{
		t.Rows[i] = RawRow(records[i+1])
	}

	return t
}

