#include "gpuMD5.h"
#include "cutil.h"

// Forward declarations of basic MD5 functions
__device__ UINT F(UINT x, UINT y, UINT z);
__device__ UINT G(UINT x, UINT y, UINT z);
__device__ UINT H(UINT& x, UINT& y, UINT& z);
__device__ UINT I(UINT& x, UINT& y, UINT& z);

__device__ UINT ROTATE_LEFT(UINT& x, UINT& n);

__device__ void FF(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void GG(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void HH(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);
__device__ void II(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac);

// F, G and H are basic MD5 functions: selection, majority, parity 
__device__ UINT F(UINT x, UINT y, UINT z){
  return ((x & y) | (~x & z));
}
__device__ UINT G(UINT x, UINT y, UINT z){
  return ((x & z) | (y & ~z));
}
__device__ UINT H(UINT& x, UINT& y, UINT& z){
  return (x ^ y ^ z);
}
__device__ UINT I(UINT& x, UINT& y, UINT& z){
  return (y ^ (x | ~z));
}

// ROTATE_LEFT rotates x left n bits 
__device__ UINT ROTATE_LEFT(UINT& x, UINT& n){
  return ((x << n) | (x >> (32 - n)));
}

// FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4
// Rotation is separate from addition to prevent recomputation 
__device__ void FF(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){
  a += F(b, c, d) + x + ac; 
  a =  ROTATE_LEFT(a, s); 
  a += b;
}
__device__ void GG(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){
  a += G(b, c, d) + x + ac; 
  a =  ROTATE_LEFT(a, s); 
  a += b;
}
__device__ void HH(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){  
  (a) += H ((b), (c), (d)) + (x) + (UINT)(ac); 
   (a) = ROTATE_LEFT ((a), (s)); 
   (a) += (b); 
}
__device__ void II(UINT& a, UINT& b, UINT& c, UINT& d, UINT& x, UINT s, UINT ac){ 
(a) += I ((b), (c), (d)) + (x) + (UINT)(ac); 
   (a) = ROTATE_LEFT ((a), (s)); 
   (a) += (b);
}
// CONSTANT DECLARATIONS
extern __shared__ char array[]; 

__constant__ UINT deviceTarget[4];
__device__ uint4 foundKey;

__global__ void md5Hash(UCHAR*, int*,int*, uint4*);
__device__ void pad(uint4*, int*,UINT*);
__device__ void transform(uint4* message, UINT* in, uint4* digest);


uint4 convertKey(const char* key) {

  unsigned int *word1 = (unsigned int*) &key[0];
  unsigned int *word2 = (unsigned int*) &key[4];
  unsigned int *word3 = (unsigned int*) &key[8];
  unsigned int *word4 = (unsigned int*) &key[12];

  return make_uint4(*word1,*word2,*word3,*word4);
}

void convertUint4(uint4 in, char* out) {
  
  char* word1 = (char*) &in.x;
  char* word2 = (char*) &in.y;
  char* word3 = (char*) &in.z;
  char* word4 = (char*) &in.w;
  
  for(int i =0; i<4;i++) {
   out[i] = word1[i];
  }
  for(int i =0; i<4;i++) {
   out[i+4] = word2[i];
  }
  for(int i =0; i<4;i++) {
   out[i+8] = word3[i];
  }
  for(int i =0; i<4;i++) {
   out[i+12] = word4[i];
  }
  

}
void resetResult(void) {
  uint4 zero = make_uint4(0,0,0,0);
  cudaMemcpyToSymbol(foundKey, &zero, 8);
}

void initialiseConstants(UINT* target) {
  // Copy target hash to the device (So that the comparison can be done on the GPU)
  cudaMemcpyToSymbol(deviceTarget, target, sizeof(deviceTarget));
  char zero = '0';   
  cudaMemcpyToSymbol(foundKey, &zero, sizeof(zero));
}


__device__ void padMsg(uint4* keys_d, int len, UINT* paddedMessage){ 
	
    uint4 message = *keys_d;//make_uint4(*tmp0,*tmp1,*tmp2,*tmp3);
    
  paddedMessage[0]=((message.x & 0xFF)  )| ((message.x & 0xFF00) ) |((message.x & 0x00FF0000) ) | ((message.x & 0xFF000000) );
	paddedMessage[1]=((message.y & 0xFF) )| ((message.y & 0xFF00) ) |((message.y & 0x00FF0000) ) | ((message.y & 0xFF000000) );
	paddedMessage[2]=((message.z & 0xFF) )| ((message.z & 0xFF00) ) |((message.z & 0x00FF0000) ) | ((message.z & 0xFF000000) );
	paddedMessage[3]=((message.w & 0xFF)  )| ((message.w & 0xFF00) ) |((message.w & 0x00FF0000) ) | ((message.w & 0xFF000000));
	paddedMessage[4]=0x00000000;
  paddedMessage[5]=0x00000000;
  paddedMessage[6]=0x00000000;
  paddedMessage[7]=0x00000000;
  paddedMessage[8]=0x00000000;
  paddedMessage[9]=0x00000000;
  paddedMessage[10]=0x00000000;
  paddedMessage[11]=0x00000000;
  paddedMessage[12]=0x00000000;
  paddedMessage[13]=0x00000000;
  paddedMessage[14]=len << 3;
  paddedMessage[15]=0x00000000;
 
}

// TEH KERNEL FUNCTION
__global__ void hash(uint4* keys_d, int* lengths_d, uint4* digests_d, int nKeys) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < nKeys){      
    UINT* sharedMem = (UINT*) &array;
    //UINT *in = &sharedMem[16*64];  // wrong!?
    UINT *in = sharedMem + (4 * idx);
    
    padMsg(keys_d+idx,lengths_d[idx],&in[idx*16]);
    
    transform(keys_d+idx,&in[idx*16], digests_d+idx);
  }
}

bool hashByBatch(std::vector<std::string>& keys, int batchSize) {
  foundKey = make_uint4(0,0,0,0);
  int nKeys = keys.size();
  int nBatches = 0;
  std::vector<std::string> keyBatch;
  for (int i=0;i<nKeys;i+=batchSize){    
    for (int j=nBatches*batchSize; j<(nBatches*batchSize)+batchSize && j<nKeys; ++j){
      keyBatch.push_back(keys[j]);
    }
    //printf("Calling doHash() with a batchSize of %d\n", keyBatch.size()); //DEBUG
    if (doHash(keyBatch)){
      return true;
    }
    nBatches++;
    keyBatch.clear();
  }
  return false;
}

bool doHash(std::vector<std::string>& keys) {
  using namespace std;
  
  bool success = false;
  
  // Getting the device properties.
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  //int numBlocks = deviceProp.multiProcessorCount * 2;
  //int numThreadsPerBlock = NUM_THREADS_PER_BLOCK;
  //int numThreadsPerGrid = numBlocks * numThreadsPerBlock;
  int sharedMem = deviceProp.sharedMemPerBlock / 2;
  
  //Device variables  
  uint4* keys_d;
  int* lengths_d;
  uint4* digests_d;
  
  //Host Variables  
  uint4* keys_h;
  int* lengths_h;
  int* foundKeyAddress;
  uint4 foundKey;
  //int* isFoundKey; *isFoundKey=0;
  
  //cudaHostAlloc((void**)&isKeyFound,sizeof(int),cudaHostAllocMapped);
  
  UINT permutationSize = keys.size();
  //printf("\tBatch Size: %d\n",permutationSize);
  
  //Allocate memory to device variables;
  //fprintf(stderr,"Allocating memory for device variables\n");
  cudaMalloc((void**) &keys_d, permutationSize*sizeof(uint4));
  //printf("keys_d: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &lengths_d, permutationSize*sizeof(int));
  //printf("lengths_d: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMalloc((void**) &digests_d, permutationSize*sizeof(uint4));
  //printf("digests_d: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemset(lengths_d, permutationSize*sizeof(int),0);
	
	//Allocate memory to host variables;    
  //fprintf(stderr,"Allocating memory for host variables\n");  
  keys_h = (uint4*) malloc(permutationSize*sizeof(uint4));
  lengths_h = (int*) malloc(permutationSize*sizeof(int));
//  digests_h = (uint4*) malloc(permutationSize*sizeof(uint4));  
  
  char buffer[16];
  //Copy keys in vector to host memory
  for(int i = 0;i<keys.size();i++) {  
    lengths_h[i] = keys[i].length();
  	memset(buffer,0,16);
	  memcpy(buffer,keys[i].c_str(), lengths_h[i] );
	  buffer[lengths_h[i]] =0x80;
	  keys_h[i] = convertKey( buffer );	
	
	  //char test[16];
  	//convertUint4(keys_h[i],test);
    //printf("Key %d: %s Len: %d\n",i,test, lengths_h[i]);   
   }

  //Copy Memory from host to Device
  //printf("Copying memory from host to device\n");
  cudaMemcpy((void**) keys_d, keys_h, permutationSize*sizeof(uint4),cudaMemcpyHostToDevice);
  cudaMemcpy((void**) lengths_d, lengths_h,permutationSize*sizeof(int), cudaMemcpyHostToDevice);
  //cudaMemcpy((void**) digests_d, digests_h,permutationSize*sizeof(uint4), cudaMemcpyHostToDevice);
  int NUM_KEYS = 128;
  int i;
  for(i=0; i < permutationSize; i+=NUM_KEYS){
    hash<<< 8, 16, sharedMem >>> (keys_d+i, lengths_d+i, digests_d+i, NUM_KEYS);
  }
  //hash<<< 16, 8, sharedMem >>> (keys_d+i, lengths_d+i, digests_d+i, NUM_KEYS);
  
  // Write result back
  cudaGetSymbolAddress((void**)&foundKeyAddress, "foundKey");
  cudaMemcpy((void**) &foundKey, foundKeyAddress, sizeof(uint4), cudaMemcpyDeviceToHost);
  char result[16] = {'0','0','0','0','0','0','0','0','0','0','0','0','0','0','0','0'};
  convertUint4(foundKey,result);
    
  if (result[0] == 0 || result[0] == 48){ //TODO: Fix me, I'm hacky!
    //fprintf(stderr,"Key not found.\n");
  } else {
    //printf(" * result[0] = %i|%c|%x\n", result[0]);
    fprintf(stderr,"Found key: %s\n", result);
    success = true;
  }
    
  // Reset result for later runs
  resetResult();
  
  /*
  // NOTE: Remember to uncomment the line at the end of transform() which records digest
  //Declare Result Variables  
  uint4* keys_r;
  int* lengths_r;
  uint4* digests_r;
  
  //Allocate memory to result variables
  //printf("Allocating memory to result variables\n");
  keys_r = (uint4*) malloc(permutationSize*sizeof(uint4));
  lengths_r = (int*) malloc(permutationSize*sizeof(int));
  digests_r = (uint4*) malloc(permutationSize*sizeof(uint4));
  
  memset(lengths_r,permutationSize*sizeof(int),0);
  
  //Copying memoty result variables
  //printf("Copying data from device memory to result variables\n");
  cudaMemcpy((void**) keys_r, keys_d, permutationSize*sizeof(uint4),cudaMemcpyDeviceToHost);
  cudaMemcpy((void**) lengths_r, lengths_d,permutationSize*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy((void**) digests_r, digests_d, permutationSize*sizeof(uint4),cudaMemcpyDeviceToHost);
  
    
  //Copy keys in vector to host memory
  char kr[16];
 
  for(int i = 0;i<permutationSize;i++) {
  
	
	convertUint4(keys_r[i],kr);
	printf("Key %d: %s Len: %d Digest: %08x %08x %08x %08x\n",i,kr, lengths_r[i], digests_r[i].x, digests_r[i].y,digests_r[i].z, digests_r[i].w);
   // printf("Uint4: %08x %08x %08x %08x\n", keys_r[i].x,keys_r[i].y,keys_r[i].z,keys_r[i].w);
   }
  */

  free(keys_h);
  free(lengths_h);  
  
  cudaFree(keys_d);
  cudaFree(lengths_d);
  cudaFree(digests_d);

  if (success){return true;}else{return false;}
}


__device__ void transform(uint4* message, UINT* in, uint4* digest) {

  unsigned int h0 = 0x67452301;
  unsigned int h1 = 0xEFCDAB89;
  unsigned int h2 = 0x98BADCFE;
  unsigned int h3 = 0x10325476;
  
  UINT a = h0;
  UINT b = h1;
  UINT c = h2;
  UINT d = h3;
  
  
/* Round 1 */
#define S11 7
#define S12 12
#define S13 17
#define S14 22
  FF ( a, b, c, d, in[ 0], S11, 3614090360); /* 1 */
  FF ( d, a, b, c, in[ 1], S12, 3905402710); /* 2 */
  FF ( c, d, a, b, in[ 2], S13,  606105819); /* 3 */
  FF ( b, c, d, a, in[ 3], S14, 3250441966); /* 4 */
  FF ( a, b, c, d, in[ 4], S11, 4118548399); /* 5 */
  FF ( d, a, b, c, in[ 5], S12, 1200080426); /* 6 */
  FF ( c, d, a, b, in[ 6], S13, 2821735955); /* 7 */
  FF ( b, c, d, a, in[ 7], S14, 4249261313); /* 8 */
  FF ( a, b, c, d, in[ 8], S11, 1770035416); /* 9 */
  FF ( d, a, b, c, in[ 9], S12, 2336552879); /* 10 */
  FF ( c, d, a, b, in[10], S13, 4294925233); /* 11 */
  FF ( b, c, d, a, in[11], S14, 2304563134); /* 12 */
  FF ( a, b, c, d, in[12], S11, 1804603682); /* 13 */
  FF ( d, a, b, c, in[13], S12, 4254626195); /* 14 */
  FF ( c, d, a, b, in[14], S13, 2792965006); /* 15 */
  FF ( b, c, d, a, in[15], S14, 1236535329); /* 16 */
  
  /* Round 2 */
#define S21 5
#define S22 9
#define S23 14
#define S24 20
  GG ( a, b, c, d, in[ 1], S21, 4129170786); /* 17 */
  GG ( d, a, b, c, in[ 6], S22, 3225465664); /* 18 */
  GG ( c, d, a, b, in[11], S23,  643717713); /* 19 */
  GG ( b, c, d, a, in[ 0], S24, 3921069994); /* 20 */
  GG ( a, b, c, d, in[ 5], S21, 3593408605); /* 21 */
  GG ( d, a, b, c, in[10], S22,   38016083); /* 22 */
  GG ( c, d, a, b, in[15], S23, 3634488961); /* 23 */
  GG ( b, c, d, a, in[ 4], S24, 3889429448); /* 24 */
  GG ( a, b, c, d, in[ 9], S21,  568446438); /* 25 */
  GG ( d, a, b, c, in[14], S22, 3275163606); /* 26 */
  GG ( c, d, a, b, in[ 3], S23, 4107603335); /* 27 */
  GG ( b, c, d, a, in[ 8], S24, 1163531501); /* 28 */
  GG ( a, b, c, d, in[13], S21, 2850285829); /* 29 */
  GG ( d, a, b, c, in[ 2], S22, 4243563512); /* 30 */
  GG ( c, d, a, b, in[ 7], S23, 1735328473); /* 31 */
  GG ( b, c, d, a, in[12], S24, 2368359562); /* 32 */

  /* Round 3 */
#define S31 4
#define S32 11
#define S33 16
#define S34 23
  HH ( a, b, c, d, in[ 5], S31, 4294588738); /* 33 */
  HH ( d, a, b, c, in[ 8], S32, 2272392833); /* 34 */
  HH ( c, d, a, b, in[11], S33, 1839030562); /* 35 */
  HH ( b, c, d, a, in[14], S34, 4259657740); /* 36 */
  HH ( a, b, c, d, in[ 1], S31, 2763975236); /* 37 */
  HH ( d, a, b, c, in[ 4], S32, 1272893353); /* 38 */
  HH ( c, d, a, b, in[ 7], S33, 4139469664); /* 39 */
  HH ( b, c, d, a, in[10], S34, 3200236656); /* 40 */
  HH ( a, b, c, d, in[13], S31,  681279174); /* 41 */
  HH ( d, a, b, c, in[ 0], S32, 3936430074); /* 42 */
  HH ( c, d, a, b, in[ 3], S33, 3572445317); /* 43 */
  HH ( b, c, d, a, in[ 6], S34,   76029189); /* 44 */
  HH ( a, b, c, d, in[ 9], S31, 3654602809); /* 45 */
  HH ( d, a, b, c, in[12], S32, 3873151461); /* 46 */
  HH ( c, d, a, b, in[15], S33,  530742520); /* 47 */
  HH ( b, c, d, a, in[ 2], S34, 3299628645); /* 48 */
  
    /* Round 4 */
#define S41 6
#define S42 10
#define S43 15
#define S44 21
  II ( a, b, c, d, in[ 0], S41, 4096336452); /* 49 */
  II ( d, a, b, c, in[ 7], S42, 1126891415); /* 50 */
  II ( c, d, a, b, in[14], S43, 2878612391); /* 51 */
  II ( b, c, d, a, in[ 5], S44, 4237533241); /* 52 */
  II ( a, b, c, d, in[12], S41, 1700485571); /* 53 */
  II ( d, a, b, c, in[ 3], S42, 2399980690); /* 54 */
  II ( c, d, a, b, in[10], S43, 4293915773); /* 55 */
  II ( b, c, d, a, in[ 1], S44, 2240044497); /* 56 */
  II ( a, b, c, d, in[ 8], S41, 1873313359); /* 57 */
  II ( d, a, b, c, in[15], S42, 4264355552); /* 58 */
  II ( c, d, a, b, in[ 6], S43, 2734768916); /* 59 */
  II ( b, c, d, a, in[13], S44, 1309151649); /* 60 */
  II ( a, b, c, d, in[ 4], S41, 4149444226); /* 61 */
  II ( d, a, b, c, in[11], S42, 3174756917); /* 62 */
  II ( c, d, a, b, in[ 2], S43,  718787259); /* 63 */
  II ( b, c, d, a, in[ 9], S44, 3951481745); /* 64 */
  
  a += h0;
  b += h1;
  c += h2;
  d += h3;
 
	if (deviceTarget[0] == a && deviceTarget[1] == b && deviceTarget[2] == c && deviceTarget[3] == d){
	  foundKey = *message;
	}
	
	// Un-comment line below to get digests back to host
  //*digest = make_uint4(a,b,c,d);
}


void reverseHash(uint4* hash){
  //uint4 rHash;
  //TODO
}


/*
__device__ void pad(uint4* key_d, int *msgLength, UINT* paddedMessage) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;  
  
  
  //for (i = 0; i != msgLength; ++i)
  //  m[i] = message[i];
  //for(;i<56;i++)
  //  m[i] = 0;
  
  UCHAR* message = (UCHAR*) key_d;
  message[*msgLength] = 0x80;  
   
  
  for (int i = 0; i != 14; ++i) {
      paddedMessage[i] = (UINT)message[i*4+3] << 24 |(UINT)message[i*4+2] << 16 |(UINT)message[i*4+1] << 8 | (UINT)message[i*4];
  }
  
  paddedMessage[14] = *msgLength << 3;
  paddedMessage[15] = *msgLength >> 29;
}
*/

