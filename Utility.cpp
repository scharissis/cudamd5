#include "Utility.h"

#include <cstdlib>
#include <string>
#include <vector>

#include <cuda.h>

using namespace std;

vector<UINT> prepareTarget(const string& target) {
  vector<UINT> prepared;
  
  for (int i = 0; i != target.length(); i += 8) {
    string s = target.substr(i, i + 8);
    
    reverse(s.begin(), s.end());
    
    for (string::iterator sit = s.begin(); sit != s.end(); sit += 2)
      swap(*sit, *sit+1);
      
    prepared.push_back((UINT)strtoul(s.c_str(), NULL, 16));
  }
  
  return prepared;
}
