// Interface of Permutator class.

#ifndef PERMUTATOR_H
#define PERMUTATOR_H

#include <map>
#include <string>

class Permutator {
public:
  // Constructor
  Permutator(std::string);
  
  // Generates the next permutation of a particular string.
  std::string permutate(const std::string&);

private:
  std::string charset;
  char first;
  char last;
  
  std::map<char, char> replacer;
};

#endif
