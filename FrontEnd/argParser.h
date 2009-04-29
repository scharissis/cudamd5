/*
 *  argParser.h
 *  CUDA Project
 *
 *  Created by Peter Wong on 29/04/09.
 *  Copyright 2009 University of New South Wales. All rights reserved.
 *
 */

#include <iostream>
#include <string>
using namespace std;




#ifndef BASE
#define BASE


class CArguments {
    string hash;
	int c;
public:
	
	void printHash();
	CArguments (int argc, char* argv[]);
};

#endif