/**
*
* <bf_decoding.c>
*
* @version 1.0 (September 2017)
*
* Reference ISO-C99 Implementation of LEDAkem cipher" using GCC built-ins.
*
* In alphabetical order:
*
* @author Marco Baldi <m.baldi@univpm.it>
* @author Alessandro Barenghi <alessandro.barenghi@polimi.it>
* @author Franco Chiaraluce <f.chiaraluce@univpm.it>
* @author Gerardo Pelosi <gerardo.pelosi@polimi.it>
* @author Paolo Santini <p.santini@pm.univpm.it>
*
* This code is hereby placed in the public domain.
*
* THIS SOFTWARE IS PROVIDED BY THE AUTHORS ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
* WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
* OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
* EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**/
#include "bf_decoding.h"
#include "gf2x_arith_mod_xPplusOne.h"
#include <string.h>
#include <assert.h>
#include "architecture_detect.h"
#include <immintrin.h>
#include "m128i_division.h"

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

/******************** START of functions' definitions for vector operations *************************/

static inline __m128i _mm256_extractf128i_lower_ps(__m256 a) {
   return _mm256_extractf128_si256(_mm256_castps_si256(a), 0x00);
}

static inline __m128i _mm256_extractf128i_upper_ps(__m256 a) {
   return _mm256_extractf128_si256(_mm256_castps_si256(a), 0x01);
}

static inline __m256 _mm256_rotbyte(__m256 a) {
   __m128i lowerPart = _mm_castps_si128(_mm256_extractf128_ps(a, 0x00));
   __m128i upperPart = _mm_castps_si128(_mm256_extractf128_ps(a, 0x01));

   lowerPart = _mm_rotbyte_epi32(lowerPart);
   upperPart = _mm_rotbyte_epi32(upperPart);

   __m256i mergedVec = _mm256_loadu2_m128i(upperPart, lowerPart); //insert

   return mm256_castsi256_ps(mergedVec);
}

static inline __m128i _mm_rotbyte_epi32(__m128i a) {
   // ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
   __m128i leftShifted = _mm_slli_epi32 (a, 0x08);
   __m128i rightShifted = _mm_srli_epi32 (a, (DIGIT_SIZE_b - 8));

   return _mm_or_si128(leftShifted, rightShifted);
}

static inline __m64 get_64_coeff_vector(const DIGIT poly[], const __m256 first_exponent_vector) {
   __m128i lower_exponent_vector = _mm256_extractf128i_lower_ps(first_exponent_vector);
   __m128i upper_exponent_vector = _mm256_extractf128i_upper_ps(first_exponent_vector);

   __m128i addend = __m128i _mm_set1_epi32(NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b-1);

   __m128i lowerStraightIdx = _mm_sub_epi32 (addend, lower_exponent_vector);
   __m128i upperStraightIdx = _mm_sub_epi32 (addend, upper_exponent_vector);

   unsigned int indexes[8];

   __m128i lowerDigitIdx = _mm_div_epi32(lowerStraightIdx, DIGIT_SIZE_b);
   __m128i upperDigitIdx = _mm_div_epi32(upperStraightIdx, DIGIT_SIZE_b);

   __m256i digitIdx = _mm256_loadu2_m128i (upperDigitIdx, lowerDigitIdx);

   // Store 256-bits 8x(32-bit) from digitIdx into memory
   _mm256_store_ps((float*) indexes, (__m256)digitIdx);
   // POSSO FARE DIRETTAMENTE LA STORE
   // DI LOWER E UPPER

   __m256i lsw[2];

   for(int i = 0; i <= 1; i++) {
      lsw[i] = _mm256_setr_epi64x (poly[indexes[i*4]], poly[indexes[i*4+1]], poly[indexes[i*4+2]], poly[indexes[i*4+3]]);
   }

   // da continuare




}
/******************** END of functions' definitions for vector operations *************************/

int bf_decoding(DIGIT out[], // N0 polynomials
   const POSITION_T HtrPosOnes[N0][DV],
   const POSITION_T QtrPosOnes[N0][M],
   DIGIT privateSyndrome[]  //  1 polynomial
)
{
   #if P < 64
   #error The circulant block size should exceed 64
   #endif

   /* The unsatisfied parity checks are kept as N0 vectors , each one having
   * a length rounded up to the closest multiple of the coalescing factor
   * DIGIT_SIZE_B, plus one entire digit, and with the last digit padded with zeroes.
   * This strategy allows to replicate an entire digit at the end of the computed UPCs
   * effectively removing altogether the (key-dependent) checks which would happen
   * in the correlation computation process */

   uint8_t unsatParityChecks[N0][ ROUND_UP(P,DIGIT_SIZE_B)+DIGIT_SIZE_B] = {{0}};
   /*only the two last digits of each vector must be blanked*/

   POSITION_T currQBitPos[M];  //  POSITION_T == uint32_t
   DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
   int check;
   int imax = ITERATIONS_MAX;
   unsigned int synd_corrt_vec[][2]= {SYNDROME_TRESH_LOOKUP_TABLE};

   do {
      gf2x_copy(currSyndrome, privateSyndrome);

      /* Vectorized computation of the unsatisifed parity check counts,
      * semantically equivalent to the following sequential code :
      for (int i = 0; i < N0; i++) {
      for (int valueIdx = 0; valueIdx < P; valueIdx++) {
      for(int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
      POSITION_T tmp = (HtrPosOnes[i][HtrOneIdx]+valueIdx) >= P ? (HtrPosOnes[i][HtrOneIdx]+valueIdx) -P : (HtrPosOnes[i][HtrOneIdx]+valueIdx);
      if (gf2x_get_coeff(currSyndrome, tmp))
      unsatParityChecks[i*P+valueIdx]++;
   }
}
}
*/
for (int i = 0; i < N0; i++) {
   int vectorIdx;
   /* vectorized upc computation. Coalescing factor DIGIT_SIZE_B.
   * The vectorized computation computes (coalescing factor) - P%(coalescing factor)
   * garbage unsatParityChecks at the end of the vector. Such a computation
   * is "free" as they get computed in parallel to the last P% (coalescing factor)
   * ones */
   for(vectorIdx = 0; vectorIdx < P / DIGIT_SIZE_B; vectorIdx++){
      /* Phase 1 vectorized recollection of syndrome bits */
      /* 8x vectorized syndrome bits: d_v 8-bit vectors of the syndrome
      are read and packed into a DIGIT vector vecSyndBits*/
      DIGIT vecSyndBits[(DV+DIGIT_SIZE_B-1)/DIGIT_SIZE_B] = {0};
      int vecSyndBitsWordIdx;
      for(vecSyndBitsWordIdx = 0;
         vecSyndBitsWordIdx < (DV+DIGIT_SIZE_B-1)/DIGIT_SIZE_B-1;
         vecSyndBitsWordIdx++){
            for(int vecSyndBitsElemIdx = 0;
               vecSyndBitsElemIdx < DIGIT_SIZE_B;
               vecSyndBitsElemIdx++){
                  POSITION_T tmp = HtrPosOnes[i][vecSyndBitsWordIdx*DIGIT_SIZE_B+vecSyndBitsElemIdx];
                  /*note: the position will be the one of the lowest exponent in the bitvector */
                  tmp += vectorIdx * DIGIT_SIZE_B;
                  tmp = tmp >= P ? tmp - P : tmp;
                  vecSyndBits[vecSyndBitsWordIdx] = ROTBYTE(vecSyndBits[vecSyndBitsWordIdx]);
                  vecSyndBits[vecSyndBitsWordIdx] += gf2x_get_8_coeff_vector(currSyndrome,tmp);
               }
               // 8x uint32_t of HtrPosOnes[i] are loaded in a 256-bit vector unit as single precision float
               __m256 tmpReg = _mm256_loadu_ps((float*) &HtrPosOnes[i][vecSyndBitsWordIdx*DIGIT_SIZE_B]);

               // tmpReg splitted in two 128i vectors
               __m128i lowerTmp = _mm256_extractf128i_lower_ps(tmpReg);
               __m128i upperTmp = _mm256_extractf128i_upper_ps(tmpReg);

               // semantically equivalent to tmp += vectorIdx * DIGIT_SIZE_B
               __m128i addend = _mm_set1_epi32(vectorIdx * DIGIT_SIZE_B); // vectorIdx * DIGIT_SIZE_B saved in a __m128i
               lowerTmp = _mm_add_epi32(lowerTmp, addend);
               upperTmp = _mm_add_epi32(upperTmp, addend);

               tmpReg = __m256 _mm256_setr_m128 (lowerTmp, upperTmp);

               // I want to compare each element of tmpReg with P: here I'm using a compare function that will
               // return a mask (0x1d = 29 means Greater-than-or-equal (ordered, non-signaling))
               __m256 broadcastedP = _mm256_set1_ps((float) P);
               __m256 cmpMask = _mm256_cmp_ps(tmpReg, broadcastedP, 0x1d);

               // semantically equivalent to: tmp = tmp >= P ? tmp - P : tmp;
               tmpReg = _mm256_blendv_ps (tmpReg, _mm256_sub_ps(tmpReg, broadcastedP), cmpMask);

            }

            for(int vecSyndBitsElemIdx = 0;
               vecSyndBitsElemIdx < DV % DIGIT_SIZE_B;
               vecSyndBitsElemIdx++){
                  POSITION_T tmp = HtrPosOnes[i][vecSyndBitsWordIdx*DIGIT_SIZE_B+vecSyndBitsElemIdx];
                  /*note: the position will be the one of the lowest exponent in the bitvector */
                  tmp += vectorIdx * DIGIT_SIZE_B;
                  tmp = tmp >= P ? tmp - P : tmp;
                  vecSyndBits[vecSyndBitsWordIdx]  = ROTBYTE(vecSyndBits[vecSyndBitsWordIdx]);
                  vecSyndBits[vecSyndBitsWordIdx] += gf2x_get_8_coeff_vector(currSyndrome,tmp);
               }
               /*Phase 2: vectorized computation of DIGIT_SIZE_B UPCs vectorized in upcVec*/
               DIGIT upcVec = 0;
               for(int vecElem=0; vecElem < DIGIT_SIZE_B; vecElem++) {
                  DIGIT upcDigit = 0;
                  for(vecSyndBitsWordIdx = 0; vecSyndBitsWordIdx < (DV+DIGIT_SIZE_B-1)/DIGIT_SIZE_B; vecSyndBitsWordIdx++){
                     DIGIT synVec = vecSyndBits[vecSyndBitsWordIdx];
                     upcDigit += __builtin_popcountll( (unsigned long long int) (synVec & 0x0101010101010101ULL));
                     vecSyndBits[vecSyndBitsWordIdx]= synVec >> 1;
                  }
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                  upcVec |= upcDigit << (8* (vecElem) );
#else
                  upcVec |= upcDigit << (8* (7-vecElem) );
#endif
               }
               *( (DIGIT*) (unsatParityChecks+i*P + vectorIdx*DIGIT_SIZE_B )) = upcVec;
            }
         }

         /* padding unsatisfiedParityChecks */
         for (int i = 0; i < N0; i++) {
            for(int j = 0; j < DIGIT_SIZE_B; j++)
            unsatParityChecks[i][P+j] = unsatParityChecks[i][j];
         }


         // computation of syndrome weight and threshold determination
         int syndrome_wt = population_count(currSyndrome);
         int min_idx=0;
         int max_idx;
         max_idx = sizeof(synd_corrt_vec)/(2*sizeof(unsigned int)) - 1;
         int thresh_table_idx = (min_idx + max_idx)/2;
         while(min_idx< max_idx) {
            if (synd_corrt_vec[thresh_table_idx][0] <= syndrome_wt) {
               min_idx = thresh_table_idx +1;
            } else {
               max_idx = thresh_table_idx -1;
            }
            thresh_table_idx = (min_idx +max_idx)/2;
         }
         int corrt_syndrome_based=synd_corrt_vec[thresh_table_idx][1];

         //Computation of correlation  with a full Q matrix
         /* BEGINNING ORIGINAL */
         //             for (int blockIdx = 0; blockIdx < N0; blockIdx++) {
         //                endQblockIdx += qBlockWeights[blockIdx][i];
         //                int currblockoffset = blockIdx*P;
         //                for (; currQoneIdx < endQblockIdx; currQoneIdx++) {
         //                   int tmp = QtrPosOnes[i][currQoneIdx]+j; /*this must turn into a vector load of 8x UPCs*/
         //                   tmp = tmp >= P ? tmp - P : tmp;  /* this will need to be checked before and have a split vector load */
         //                   currQBitPos[currQoneIdx] = tmp;  /* only a single base index must be kept per vector load*/
         //                   currQBlkPos[currQoneIdx] = blockIdx; /* this is simply a vector filled with the block-index of the block circulant position
         //                                                         * it should be a compile time, parameter dependant constant, not a runtime computed vector! */
         //                   correlation += unsatParityChecks[tmp + currblockoffset];
         //                }
         //             }
         /* END ORIGINAL */


         for (int i = 0; i < N0; i++) {
            uint8_t qBlockIdxes[M];
            uint16_t qBlockOffsets[M];
            uint8_t qBlockIdxesCursor = 0, linearized_cursor = 0;
            int endQblockIdx = 0;
            for (int blockIdx = 0; blockIdx < N0; blockIdx++) {
               endQblockIdx += qBlockWeights[blockIdx][i];
               for (; linearized_cursor < endQblockIdx; linearized_cursor++) {
                  qBlockIdxes[linearized_cursor] = qBlockIdxesCursor;
                  qBlockOffsets[linearized_cursor] =  (uint16_t) qBlockIdxesCursor;
               }
               qBlockIdxesCursor++;
            }
            for (int j = 0; j <= P; j+=DIGIT_SIZE_B) {
               int currQoneIdx = 0; // position in the column of QtrPosOnes[][...]
               /* a vector of (single byte wide) correlation values is computed
               * at each iteration of the outer loop. 8x vectorization is possible
               * with uint64_t. The most significant byte in the uint64_t vector
               * of uint8_t is the one matching the outer loop index position*/
               uint64_t correlation = 0;

               /* a single correlation value is the sum of m "block circulant positioned" values of UPC */
               /* I need a 8x storage of UPCs which I can load as 8x bytes from upc vector starting from
               * the block circulant position.
               * A trivial implementation will thus have m variables containing 8 consecutive UPCs.
               *
               * A more reasonable one uses a single variable which gets filled
               * with 8x UPCs per shot, bar circulant cases related to 8 consecutive correlation
               * computations, and a single circulant "one" from Q (ideal opts will unroll).
               * Since the outer loop (j indexed, at line 183) is vectorized, a trailer of the last P%DIGIT_SIZE_b should be added
               * (or not: a single vector computation discarding part of the results in correlation is equally fine) */

               /* This vectorization strategy will benefit from having a vector with the block circulant pos.s in Q
               * and another one with the block idxes. Both can be vector packed: pos.s are 16 bit wide, block idx.es are 2b wide
               * FTTB I will generate both of them at runtime, but the latter can be generated as a #define.*/

               for (currQoneIdx = 0; currQoneIdx < M; currQoneIdx++) {
                  int tmp = QtrPosOnes[i][currQoneIdx]+j; /* this must turn into a vector load of 8x UPCs with consecutive j indices */
                  tmp = tmp >= P ? tmp - P : tmp;  /* this will need to be checked before and have a split vector load */
                  currQBitPos[currQoneIdx] = tmp;  /* only a single base index must be kept per vector load*/
                  //                 for(int delta_j = 0 ; delta_j < DIGIT_SIZE_B ; delta_j++ ){
                  //                   correlation_contrib =  (correlation_contrib << 8) | (unsatParityChecks[qBlockOffsets[currQoneIdx]][tmp+delta_j]);
                  //                 }
                  correlation += *(uint64_t*)(&unsatParityChecks[qBlockOffsets[currQoneIdx]][tmp]);
               }

               /* Correlation based flipping */
               for (int delta_j = 0 ; delta_j < DIGIT_SIZE_B &&(j+delta_j < P) ; delta_j++){
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                  uint64_t single_correl = ( correlation >> (8*delta_j) ) & 0xFF;
#else
                  uint64_t single_correl = ( correlation >> (8*(DIGIT_SIZE_B-1-delta_j)) ) & 0xFF;
#endif
                  if ((single_correl > corrt_syndrome_based)) { /* correlation will always be null for garbage bits */
                     gf2x_toggle_coeff(out+NUM_DIGITS_GF2X_ELEMENT*i, (j+delta_j)%P); /*this can be vectorized if a vector of correlations are available*/
                     for (int v = 0; v < M; v++) { /* the same goes here */
                        unsigned syndromePosToFlip;
                        for (int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
                           syndromePosToFlip = (HtrPosOnes[qBlockIdxes[v]][HtrOneIdx] + (currQBitPos[v]+delta_j)%P );
                           syndromePosToFlip = syndromePosToFlip >= P ? syndromePosToFlip - P : syndromePosToFlip;
                           gf2x_toggle_coeff(privateSyndrome, syndromePosToFlip);
                        }
                     } // end for v
                  } // end if
               }
            } // end for j
         } // end for i

         imax = imax - 1;
         check = 0;
         while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);

      } while (imax != 0 && check < NUM_DIGITS_GF2X_ELEMENT);

      return (check == NUM_DIGITS_GF2X_ELEMENT);
   }  // end QdecodeSyndromeThresh_bitFlip_sparse
