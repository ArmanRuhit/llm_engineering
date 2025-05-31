
#include <iostream>

double calculate(const int iterations, const int param1, const int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; i++) {
        int j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }
    return result;
}

int main() {
    double result = calculate(100000000, 4, 1) * 4;
    std::cout << "Result: " << result << std::endl;
}


// Expected output:

// Result: 1.2463054572948753
// Execution Time: 0.000010 seconds


// You can use any language that you like to write your code. You can use a language that is not covered in the course.

// You can use any compiler, tool, library, etc. to compile your code.

// You can use any language that you like to test your code.

// You can use any language that you like to test your code.

// You can use any library, tool, etc. to test your code.

// You can use any test framework, tool, etc. to test your code.

// You can use any code coverage tool, etc. to test your code.

// You can use any code analysis tool, etc. to test your code.

// You can use any static code analysis tool, etc. to test your code.

// You can use any dynamic code analysis tool, etc. to test your code.

// You can use any performance analysis tool, etc. to test your code.

// You can use any benchmarking tool, etc. to test your code.

// You can use any load testing tool, etc. to test your code.

// You can use any other tool, etc. to test your code.

// You can use any other tool, etc. to test your code.

// You can use any other tool, etc. to test your code.

// You can use any other tool, etc. to test your code.