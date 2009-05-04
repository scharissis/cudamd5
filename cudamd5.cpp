/*
 *  cudamd5.cpp
 *  CUDA Project
 *
 *  Created by Peter Wong on 29/04/09.
 *  Copyright 2009 University of New South Wales. All rights reserved.
 *
 */

#include <string>
#include <iostream>
#include "cudamd5.h"
#include "md5test.cpp"
#include "Permutator.h"

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace std;

//static void MDTimeTrial_CPU ();
static void MDString (char *inString);
int deviceQuery();

int main(int argc, char *argv[]) {

	// Declare the supported options.
	int min,max;
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,?", "Produces this help message.")
    ("query,q", "Queries CUDA devices.")
    ("target,t", po::value<string>(), "Target digest")
    ("charset,c", po::value<string>(), "Message character set")
    ("min", po::value<int>()->default_value(1), "Minimum message string length")
    ("max", po::value<int>()->default_value(3), "Maximum message string length")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    

	// Check for required arguments
  if (vm.count("help")) {
      cout << desc << "\n";
      return 0;
  }
  
  if (vm.count("verbose")) {
  	//TODO: Print all options...
  }

   if (vm.count("query")) {
  	deviceQuery();
  }

  if (!vm.count("target")) {
    cerr << "Target hash required. [--target=HASH]" << endl;
    return 1;
  } else {
  		/*
  		if ( (vm["target"].as<string>()).size() != 32){
  			cerr << "Invalid target hash." << endl;
  			return 1;
  		}
  		*/
  }
  
  if (!vm.count("charset")) {
    cerr << "Character set required. [--charset=SET]" << endl;
    return 1;
  }
 
	// DEBUG
  #ifdef DEBUG
  cout << "Target: " << vm["target"].as<string>() << endl;
  cout << "Charset: " << vm["charset"].as<string>() << endl;
  cout << "Min: " << vm["min"].as<int>() << endl;
  #endif
  
  string charset(vm["charset"].as<string>());
  Permutator p(charset);  
  
  string message(vm["min"].as<int>(), p.first);
  string end(vm["max"].as<int>() + 1, p.first);
  
  do {
    MDString(message.c_str());
    message = p.permutate(message);
  } while (message != end);
  
  return 0;
}


