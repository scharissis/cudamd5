// CudaMD5.cpp

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include "MD5_GPU.h"
#include "Utility.h"

namespace po = boost::program_options;
using namespace std;

int main(int argc, char** argv) {
  int minLen;
  int maxLen;
  string targetDigest;
  string targetCharset;
  
  po::options_description generalOptions("General options");
  generalOptions.add_options()
    ("help,?", "prints this help message")
    ("verbose,v", "prints out extra information");
    ;
    
  po::options_description hashingOptions("Required options");
  hashingOptions.add_options()
    ("target,t", po::value<string>(&targetDigest), "target digest")
    ("min,m", po::value<int>(&minLen)->default_value(1), 
      "minimum message length")
    ("max,n", po::value<int>(&maxLen)->default_value(16), 
      "maximum message length")
    ("charset,c", po::value<string>(&targetCharset)->default_value("0123abcd"), 
      "message character set")
    ;
    
  po::options_description allOptions("Allowed options");
  allOptions.add(generalOptions).add(hashingOptions);
  
  // Parse the given program options.
  po::variables_map args;
  try {
    po::store(po::parse_command_line(argc, argv, allOptions), args);
    po::notify(args);
  } catch (exception) {}
  
  if (args.count("target")) {
    initialiseGPU(targetDigest, targetCharset);
    findMessage(minLen, maxLen, targetCharset.length());
  }
  else {
    cout << allOptions << endl;
  }

  return EXIT_SUCCESS;
}

