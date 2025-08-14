// A simple program that computes the square root of a number
#include <iostream>
#include <string>

#include "MathFunctions.h"

using namespace mathfunctions;

int main(int argc, char* argv[])
{
  if (argc < 2) {
    // report version
    std::cout << "Usage: " << argv[0] << " number" << std::endl;
    return 1;
  }

  // convert input to double
  double const inputValue = std::stod(argv[1]);

  // TODO 6: Replace sqrt with mathfunctions::sqrt

  // calculate square root
  double const outputValue = sqrt(inputValue);
  std::cout << "The square root of " << inputValue << " is " << outputValue
            << std::endl;
  return 0;
}