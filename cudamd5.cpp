#include <cstdlib>
#include <iostream>
#include <string>
#include "cuda.h"
#include "cutil.h"
#include "gpuMD5.h"
#include "constants.h"
#include <math.h>

#include <boost/lexical_cast.hpp>
#include <boost/program_options.hpp>
namespace po = boost::program_options;

#include "deviceQuery.h"
#include "md5test.cpp"
#include "Permutator.h"

using namespace std;

typedef unsigned int UINT;

UCHAR c2c (char c);

UINT findTotalPermutations(int charsetSize, int min, int max) {
  UINT total = 0;
  UINT i    = 0;
  for(i=min; i<max+1;i++)
    total += pow(charsetSize,i);
  return total;
}


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
    ("cpu", "Runs only on the CPU")
    ("target,t", po::value<string>(), "Target digest")
    ("charset,c", po::value<string>(), "Message character set")
    ("min", po::value<int>()->default_value(1), "Minimum message length")
    ("max", po::value<int>()->default_value(3), "Maximum message length")
    ("dictionary,d", "Reads words from stdin")
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
  else if (vm.count("cpu")){
      // DEBUG
      string warning = "*** Note: CPU-version is not complete. ***\nIt does not yet check for the target hash; it runs through all keys and digests them.\n\n";
      std::cout << warning << std::endl;
      
      string targetHash = vm["target"].as<string>();
      minLen = vm["min"].as<int>();
      maxLen = vm["max"].as<int>();
      string charset(vm["charset"].as<string>());
      Permutator p(charset);  
        
      string message(minLen, p.first);
      string end(maxLen + 1, p.first);
      
      int nKeys=0;
      do {
          MDCheck(message.c_str());
          message = p.permutate(message);
          ++nKeys;
      } while (message != end);
      printf("Searched %d keys.\n", nKeys);
   
  } else if (vm.count("target")) {    
      isVerbose = vm.count("verbose");
      minLen = vm["min"].as<int>();
      maxLen = vm["max"].as<int>();
      if (maxLen > 15) { fprintf(stderr,"Maximum length is 15\n"); exit(1); }
      assert(maxLen < 16);
        
      if (isVerbose) {
        cout << "Target digest: " << vm["target"].as<string>() << endl;
        cout << "Message character set: " << vm["charset"].as<string>() << endl;
        cout << "Minimum message length: " <<  minLen << endl;
        cout << "Maximum message length: " <<  maxLen << endl << endl;
      }

      // Getting the device properties.
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      int numBlocks = deviceProp.multiProcessorCount * 2;
      int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
      int numThreadsPerGrid = numBlocks * numThreadsPerBlock;
      int sharedMem = deviceProp.sharedMemPerBlock / 2;
        
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
      initialiseConstants(reversedTargetHash);

      vector<string> messages;
      messages.reserve(2000000);
        
	  if (isVerbose){
          printf("Reversed target hash: %08x %08x %08x %08x\n\n", reversedTargetHash[0], reversedTargetHash[1],reversedTargetHash[2],reversedTargetHash[3]);
	  }
        
      if (vm.count("charset")){
          string charset(vm["charset"].as<string>());
          Permutator p(charset);  

          string message(minLen, p.first);
          string end(maxLen + 1, p.first);

          UINT totalPermutations = findTotalPermutations(charset.length(),minLen,maxLen);
      
          
          // Populate messages vector with all possible permutations 
          for (int i = 0; i < totalPermutations; i++) {
            messages.push_back(message);
            message = p.permutate(message);
          }
          printf("%d permutations generated.\nSearching...\n", messages.size());    //DEBUG    
          if (!hashByBatch(messages,2000000)){
            fprintf(stderr,"Key not found.\n");
          }
          printf("%d keys searched.\n", messages.size());
      } else {
	      std::cout << "Loading words from stdin ...\n";
	      std::string word;
	      while(std::cin >> word)
	      {
	        if (word.size() < 16){
      		  messages.push_back(word);
      		}
	      }
	      std::cout << "Loaded " << messages.size() << " words.\n\n";
	     
	
	      if (!hashByBatch(messages,2000000)){
	        fprintf(stderr,"Key not found.\n");
	      }
	      
	      std::cout << "Searched " << messages.size() << " words." << std::endl;
	  }
        
  } else {
    cout << desc << endl;
  }
  
  return EXIT_SUCCESS;
}



UCHAR c2c (char c){
  return (UCHAR)((c > '9') ? (c - 'a' + 10) : (c - '0'));
}


