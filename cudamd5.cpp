// Main program.

#include <cstdlib>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>

#include "md5GPU.h"
#include "Permutator.h"
#include "Utility.h"

namespace po = boost::program_options;

using namespace std;

int main(int argc, char** argv) {
  bool isVerbose = false;
  int minLen = 1;
  int maxLen = 3;
  string target = "";
  string charset = "";
  
	// Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,?", "Produces this help message")
    ("query,q", "Prints out properties of CUDA devices")
    ("verbose,v", "Turns on verbose mode")
    ("target,t", po::value<string>(), "Target digest")
    ("charset,c", po::value<string>(), "Message character set")
    ("min", po::value<int>()->default_value(1), "Minimum message length")
    ("max", po::value<int>()->default_value(3), "Maximum message length")
  ;
  
  // Parse the given program options.
  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (exception) {}
  
  // If --help was given, print out the options and quit.
  if (vm.count("help")) {
    cout << desc << endl;
    return EXIT_SUCCESS;
  }
  
  // If --query was given, run deviceQuery and quit.
  if (vm.count("query")) {
    //deviceQuery();
    return EXIT_SUCCESS;
  }
  
  // Otherwise, program expects at least --target and --charset.
  if (!vm.count("target") | !vm.count("charset")) {
    cout << desc << endl;
    return EXIT_SUCCESS;
  }
  
  // Collect the arguments.
  isVerbose = vm.count("verbose");
  minLen = vm["min"].as<int>();
  maxLen = vm["max"].as<int>();
  target = vm["target"].as<string>();
  charset = vm["charset"].as<string>();
  
  // If verbose mode, print out the arguments.
  if (isVerbose) {
    cout << "Target digest: " << target << endl;
    cout << "Message character set: " << charset << endl;
    cout << "Minimum message length: " <<  minLen << endl;
    cout << "Maximum message length: " <<  maxLen << endl;
  }
  
  

  // Create our password permutator.
  Permutator p(charset);  
  
  // Generate the starting password.
  string startPass(minLen, p.first);
  
  // Generate the one-past-last password to check.
  string endPass(maxLen + 1, p.first);
  
  
  for (string currentPass = startPass; currentPass != endPass;
    currentPass = p.permutate(currentPass)) {
    
    
  }
  
  return EXIT_SUCCESS;
}
