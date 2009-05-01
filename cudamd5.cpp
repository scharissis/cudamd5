/*
 *  cudamd5.cpp
 *  CUDA Project
 *
 *  Created by Peter Wong on 29/04/09.
 *  Copyright 2009 University of New South Wales. All rights reserved.
 *
 */

#include <string>
#include "cudamd5.h"
#include "md5test.cpp"
#include "Permutator.h"

//static void MDTimeTrial_CPU ();
static void MDString (char *inString);


int main(int argc, char *argv[]) {

	
	Permutator p("0123456789abcdefghijklmnopqrstuvwxyz");
	parseArg(argc, argv);
	std::string s("0000000000");
	
	for (int i = 0; i != 1000000; ++i) {
	MDString((s.c_str()));
		//fprintf(stderr,"Counter: %d\n",i);	
	s = p.permutate(s);
		
	}

	/*End of program*/
	return 0;
}


