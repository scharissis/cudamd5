/*
 *  argParser.cpp
 *  CUDA Project
 *
 *  Created by Peter Wong on 29/04/09.
 *  Copyright 2009 University of New South Wales. All rights reserved.
 *
 */

#include "argParser.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <ctype.h>
#include <string>

#define DEBUG 1


int parseArg(int argc, char* argv[]) {
	
	int c;
	//int opterr = 0;
	while ((c = getopt (argc, argv, "t:s:n:v:b:q")) != -1) {
		switch (c) {
			case 't':
#ifdef DEBUG
				fprintf(stderr,"Target Hash: %s\n",optarg);
#endif
				
				break;
			case 'h':
				//	printUsage();
				break;
			case 's':
#ifdef DEBUG
				fprintf(stderr,"Start Character Set: %s\n",optarg);				
#endif
				break;
			
			case 'v':
				//	printVersion();
				break;

			case 'b':
				//	MDTimeTrial_CPU ();
				//	TODO: This is in md5test.cpp but I can't get it to work here
				break;
			case 'q':
				deviceQuery();
				break;
			case '?':
				if (optopt == 'H' || optopt == 'l' || optopt == 'n') {
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				} else if (isprint (optopt)) {
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				} else {
					fprintf (stderr,
							 "Unknown option character `\\x%x'.\n",
							 optopt);
				}
			default:
				abort ();
		}     		
	
	}

	return 0;
}
