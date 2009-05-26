#include <cstdlib>
#include <iostream>
#include <string>
#include "cuda.h"
#include "gpuMD5.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "deviceQuery.h"
#include "md5test.cpp"
#include "Permutator.h"
#include "arrayTest.h"

using namespace std;

typedef unsigned int UINT;

//static void MDTimeTrial_CPU ();
//static void MDString (char *inString);

int main(int argc, char *argv[]) {
  bool isVerbose = false;
  int minLen;
  int maxLen;
  string message;
  
	// Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,?", "Produces this help message")
    ("query,q", "Checks system for CUDA compatibility")
    ("verbose,v", "Turns on verbose mode")
    ("target,t", po::value<string>(), "Target digest")
    ("charset,c", po::value<string>(), "Message character set")
    ("min", po::value<int>()->default_value(1), "Minimum message length")
    ("max", po::value<int>()->default_value(3), "Maximum message length")
  ;

  po::variables_map vm;
  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
  } catch (exception) {}

  if (vm.count("help")) {
    cout << desc << endl;
  }
  else if (vm.count("query")) {
    deviceQuery();
  }
  else if (vm.count("target") && vm.count("charset")) {    
    isVerbose = vm.count("verbose");
    minLen = vm["min"].as<int>();
    maxLen = vm["max"].as<int>();
    
    if (isVerbose) {
      cout << "Target digest: " << vm["target"].as<string>() << endl;
      cout << "Message character set: " << vm["charset"].as<string>() << endl;
      cout << "Minimum message length: " <<  minLen << endl;
      cout << "Maximum message length: " <<  maxLen << endl;
    }
    
    string charset(vm["charset"].as<string>());
    Permutator p(charset);  
    
    string message(minLen, p.first);
    string end(maxLen + 1, p.first);
    
    /*
    do {
      MDString(message.c_str());
      message = p.permutate(message);
    } while (message != end);
    */
    
    UINT* h[16];
    UINT* d;
    
    cudaMalloc((void **)&d, 16 * sizeof(UINT));
    
    int a = 4;
    int b = 3;
    md5Hash <<< a, b >>> (d);
    
    cudaMemcpy(h, d, 64, cudaMemcpyDeviceToHost);
    
      int i;
  for (i = 0; i != 16; ++i)
    printf("%2d %08x\n", i, h[i]);
    
    cudaFree(d);
  }
  else {
    cout << desc << endl;
  }
  
  return EXIT_SUCCESS;
}


