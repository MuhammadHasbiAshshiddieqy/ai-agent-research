package main

import "fmt"

func main() {
	fmt.Println("COBA")
	tes := coba_aja(2)
	Uhuy()
	fmt.Println(tes)
	for {
		fmt.Println("YYY")
		break
	}
}

type tes struct {
	id   int
	name string
}

// TableName adheres to GORM Model interface to define the table name of the model.
func (t tes) TableName() string {
	return "apa"
}

func coba_aja(i int) int {
	return i
}

func Uhuy() int {
	return 1
}
