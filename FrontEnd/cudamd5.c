/*
 *  cudamd5.c
 *  CUDA Project
 *
 *  Created by Peter Wong on 29/04/09.
 *  Copyright 2009 University of New South Wales. All rights reserved.
 *
 */

#include "cudamd5.h"

#include "argParser.h"


int main(int argc, char *argv[]) {


	//parseArg(argc, argv);
	
	CArguments args (argc,argv);
	args.printHash();
	
	/*End of program*/
	return 0;
}