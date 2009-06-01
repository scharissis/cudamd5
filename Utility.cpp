#include "Utility.h"

#include <string>
#include <vector>

#include <cuda.h>

using namespace std;

uint4 prepareTarget(const string& target) {
  vector<uint1> parts(4);
  
  for (int i = 0; i != target.length(); i += 8) {
    string s = target.substr(i, i + 8);
    
    reverse(s.begin(), s.end());
  }
}