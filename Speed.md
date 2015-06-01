# Application Speed #

---



&lt;BR&gt;



**May 31st:**

&lt;BR&gt;


stefano@stefano-pc:~/CUDA$ time ./cudamd5 -cabcde -t4e510be093d346512011c3f4fe36e4af --min=1 --max=6
Hash found: bababa
4608 keys searched.


&lt;BR&gt;



real    0m1.032s

&lt;BR&gt;


user    0m0.840s

&lt;BR&gt;


sys     0m0.184s

&lt;BR&gt;




&lt;BR&gt;


**Throughput =~ 4,000 hashes/second.**


Amazingly slow... We suspect it's an issue with memory coalescing.

---


**June 9th, [revision 67](https://code.google.com/p/cudamd5/source/detail?r=67):**

&lt;BR&gt;


Possible permutations: 1,679,620

&lt;BR&gt;


Our program: 3.54,3.55,3.56,3.58,3.56 (seconds) --> 3.54s Average

&lt;BR&gt;


Mario's    : 3.40,3.46,3.37,3.30,3.35 (seconds) --> 3.40s Average

&lt;BR&gt;



stefano@stefano-pc:~/Desktop/cudamd5$ time ./cudamd5 -cabcdefghijklmnopqrstuvwxyz -t01483e083bee3dc729fe1daa7df2fa42 --min=1 --max=5

&lt;BR&gt;


Possible permutations:12,356,630


&lt;BR&gt;



real	0m12.132s

&lt;BR&gt;


user	0m11.353s

&lt;BR&gt;


sys	0m0.776s

&lt;BR&gt;



We re-ran this test 5 times:

&lt;BR&gt;


real	0m12.132s

&lt;BR&gt;


real	0m12.216s

&lt;BR&gt;


real	0m12.165s

&lt;BR&gt;


real	0m12.127s

&lt;BR&gt;


real	0m12.203s

&lt;BR&gt;


Average = 12.169

&lt;BR&gt;


**Throughput =~ 1 015 452.07 hashes/second.**

stefano@stefano-pc:~/Desktop/cudamd5$ time ./cudamd5 -cabcdefghijklmnopqrstuvwxyz0123 -t99b5fb49d3934361b08d3a548d055471 --min=1 --max=5
Possible permutations:25,137,930


&lt;BR&gt;



real	0m24.055s

&lt;BR&gt;


user	0m22.737s

&lt;BR&gt;


sys	0m1.300s

&lt;BR&gt;




&lt;BR&gt;


**Throughput =~ 1 045 018.91 hashes/second.**

**June 10th, [revision 71](https://code.google.com/p/cudamd5/source/detail?r=71):**

&lt;BR&gt;


I decided to do a larger test: All lowercase letters and numbers up to a password size of 5:

&lt;BR&gt;



stefano@stefano-pc:~/Desktop/cudamd5$ time ./cudamd5 -c0123456789abcdefghijklmnopqrstuvwxyz -t95ebc3c7b3b9f1d2c40fec14415d3cb8 --min=1 --max=5 

&lt;BR&gt;


Found key: zzzzz� 

&lt;BR&gt;


62193780 keys searched.

&lt;BR&gt;



real	4m3.933s

&lt;BR&gt;


user	0m57.560s

&lt;BR&gt;


sys	0m8.993s

&lt;BR&gt;


It took 30 seconds to initialise the 62M string vector, 2mins to do the hashing/searching/cracking and then it seemed to do nothing for 1min33s ... strange. I don't yet know why.

However, assuming 2mins to search ~62M hashes:

&lt;BR&gt;


**Throughput =~ 518 281.5 hashes/second.**

---

**June 11th, [revision 71](https://code.google.com/p/cudamd5/source/detail?r=71):**

&lt;BR&gt;


I did some testing; I compared vectors to optimised vectors and arrays, in terms of population time. These were my results:

&lt;BR&gt;



The size of the input was 8,000,000 strings of length 0 < length < 17.

&lt;BR&gt;


The optimised vector used 'vector.reserve(8000000)'.

&lt;BR&gt;


I couldn't test an array, as I couldn't increase my stack size to a large enough size (32MB). I'll try again later.

| Vector | O. Vector | Array |
|:-------|:----------|:------|
| 23.9   | 15.3      | N/A   |
| 16.2   | 15.4      | N/A   |
| 16.3   | 15.2      | N/A   |
| 21.5   | 15.3      | N/A   |
| 16.2   | 15.3      | N/A   |


---


**June 13th, [revision 72](https://code.google.com/p/cudamd5/source/detail?r=72):**

&lt;BR&gt;



I fixed the issue that I discovered on June 9th; host memory was not being free()'d. I re-ran the same test, which took ~4mins, and these were my results:

&lt;BR&gt;



stefano@stefano-pc:~/Desktop/cudamd5$ time ./cudamd5 -c0123456789abcdefghijklmnopqrstuvwxyz -t95ebc3c7b3b9f1d2c40fec14415d3cb8 --min=1 --max=5


&lt;BR&gt;

Found key: zzzzz�

&lt;BR&gt;


62193780 keys searched.

&lt;BR&gt;



real    0m55.603s

&lt;BR&gt;


user    0m52.683s

&lt;BR&gt;


sys     0m2.924s

&lt;BR&gt;



Key generation took approximately 29seconds.

**Throughput =~ 1,130,796 hashes/second.** (With key generation time)

&lt;BR&gt;


**Throughput =~ 2,392,068 hashes/second.** (Without key generation time)


---


**June 19th, [revision 83](https://code.google.com/p/cudamd5/source/detail?r=83):**

&lt;BR&gt;


This is the 4th version of our code, as a result of a major re-write. Now, after having moved the key generation/permutation to the GPU, almost everything occurs on the GPU!
```
./cudamd5 --min=1 --max=5 -c=-*%^abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -t8f113e38d28a79a5a451b16048cc2b72
Found Key: ZZZZZ
```
Average time taken (5 runs): 27.272 seconds

1,370,581,547 were searched in 27.272 seconds which gives us a speed of:

&lt;BR&gt;


**Throughput =~ 50,255,997 hashes/second.**

Update: Ran this exactly, but on a C2050, and it completed in 2.543 seconds. 

&lt;BR&gt;


**Throughput =~ 137M hash/sec.**