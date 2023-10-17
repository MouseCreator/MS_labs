import calculator as calc
import reader as io

input =  io.read_image('images/x2.bmp')
ouput = io.read_image('images/y5.bmp')
method = "G"

calc_operator = calc.calculate(input, ouput, method)

result = calc.to_test(calc_operator, input, ouput)

io.write_image('images/result.bmp', result)