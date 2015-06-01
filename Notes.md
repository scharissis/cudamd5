# Plan #

  1. Write the permutation function.
    * Given a string, generate the next permutation.
    * Increment the right-most char, then roll over when necessary.
    * OR Partition the domain into n sections.

  1. Start with a simple solution.

  1. Implement a sequential solution first in C for comparison purposes.



bba = b00 + ba
> =


---

  1. Main CUDA `__global__ function - md5Hash(char* message, unit* digest(4))`
  1. `__device__ void prepare(char* message, uint* message (16x))`
  1. `__device__ void hash(uint* message (16x), uint* digest(4))`
    * hash divides the message into 64 byte chunks and calls transform() on each chunk.
  1. `__device__ void transform(uint* chunk (16), uint* digest (4))`
    * transform performs magic on the chunk and puts result into digest.
Once hash finishes, digest will contain the final MD5 digest.