#include <cstdlib>
#include <iostream>
#include <string>
#include "cuda.h"
#include "cutil.h"
#include "gpuMD5.h"
#include "constants.h"

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "deviceQuery.h"
#include "md5test.cpp"
#include "Permutator.h"

using namespace std;

typedef unsigned int UINT;

UCHAR c2c (char c);

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
    
    // Getting the device properties.
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int numBlocks = deviceProp.multiProcessorCount * 2;
    int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
    int numThreadsPerGrid = numBlocks * numThreadsPerBlock;
    int sharedMem = deviceProp.sharedMemPerBlock / 2;
    
    //UINT target[4];
    string targetHash = vm["target"].as<string>();
    
    // Reverse target endianess
    UINT reversedTargetHash[4];
    for (int c=0;c<targetHash.size();c+=8) {
      UINT x = c2c(targetHash[c]) <<4 | c2c(targetHash[c+1]); 
      UINT y = c2c(targetHash[c+2]) << 4 | c2c(targetHash[c+3]);
      UINT z = c2c(targetHash[c+4]) << 4 | c2c(targetHash[c+5]);
      UINT w = c2c(targetHash[c+6]) << 4 | c2c(targetHash[c+7]);
      reversedTargetHash[c/8] = w << 24 | z << 16 | y << 8 | x;
    }
    //target = make_uint4(reversedTargetHash[0], reversedTargetHash[1], reversedTargetHash[2], reversedTargetHash[3]);
    //printf("Target (Reversed) Hash: %08x %08x %08x %08x\n", reversedTargetHash[0], reversedTargetHash[1], reversedTargetHash[2], reversedTargetHash[3]);
    initialiseConstants(reversedTargetHash);

    int nKeys=numThreadsPerGrid;
    printf("Testing with chunks of size %d\n", numThreadsPerGrid);
    vector<string> messages;
    while(true){
      for (int i = 0; i != numThreadsPerGrid; ++i) {
        messages.push_back(message);
        message = p.permutate(message);
      }
     
      if(doHash(messages))
        break;
      nKeys+=numThreadsPerGrid;
       if(message.length() > maxLen)
        break;
      
      messages.clear();
     }
    printf("%d keys searched.\n", nKeys);

        
  }
  else {
    cout << desc << endl;
  }
  
  return EXIT_SUCCESS;
}

UCHAR c2c (char c){
  return (UCHAR)((c > '9') ? (c - 'a' + 10) : (c - '0'));
}


