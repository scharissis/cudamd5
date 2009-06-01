// Implementation of Permutator class.

#include <utility>

#include "Permutator.h"

using std::make_pair;
using std::string;

Permutator::Permutator(const std::string& charset) :
  charset(charset) {
  char key;
  char value;
  
  // Remember the first and last characters.
  first = charset[0];
  last = charset[charset.size() - 1];
  
  // Populate the replacer map.
  string::const_iterator it = charset.begin();
  string::const_iterator end = --charset.end();
  while (it != end) {
    // Annoyingly you cannot move an iterator inside the pair() constructor,
    // so we need to copy out the value and increment the iterator outside.
    key = *it;
    value = *++it;
    
    replacer.insert(make_pair(key, value));
  }
}

string Permutator::permutate(const std::string& s) {
  string result(s);
  string::iterator it = result.begin();
  string::iterator end = result.end();
  bool incrementNext = true;
  
  while (incrementNext) {
    // If the current character isn't the "last", then we just increment it.
    if (*it != last) {
      incrementNext = false;
      *it = replacer[*it];
    }
    // Otherwise, we need to roll over and increment the next character.
    else {
      *it++ = first;
      
      // If we are at the end, append the first character on the end to roll
      // over, and we can terminate the loop earlier.
      if (it == end) {
        result.push_back(first);
        //end = result.end(); // Unnecessary.
        incrementNext = false;
      }
    }
  }
  
  return result;
}
