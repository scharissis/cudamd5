// Interface of Permutator class.

#ifndef PERMUTATOR_H
#define PERMUTATOR_H

#include <string>

#include <boost/unordered_map.hpp>

class Permutator {
public:
  // Constructor
  Permutator(const std::string&);
  
  // Generates the next permutation of a particular string.
  std::string permutate(const std::string&);
  
  char first;
  char last;

private:
  std::string charset;  
  boost::unordered_map<char, char> replacer;
};

#endif
