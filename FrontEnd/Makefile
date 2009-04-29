CC = g++ -Wall -Werror -O

all : cudamd5

clean :
	rm -f cudamd5  *.o
	
debug:	dsm test




cudamd5 : cudamd5.o argParser.o
	$(CC) -o $@ $^





cudamd5.o : cudamd5.c
	$(CC) -c $?

argParser.o : argParser.c
	$(CC) -c $?
