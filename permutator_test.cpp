#include <cstdlib>
#include <iostream>

#include "Permutator.h"

using namespace std;

int main(int argc, char* argv[]) {
  Permutator p("0123456789abcdefghijklmnopqrstuvwxyz");
  
  cout << p.permutate("z0") << endl;

  return EXIT_SUCCESS;
}
