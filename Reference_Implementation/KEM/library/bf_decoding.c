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

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )

int bf_decoding(DIGIT out[], // N0 polynomials
                const POSITION_T HtrPosOnes[N0][DV],
                const POSITION_T QtrPosOnes[N0][M],
                DIGIT privateSyndrome[]  //  1 polynomial
               )
{
#if P < 64
#error The circulant block size should exceed 64
#endif

   uint8_t unsatParityChecks[N0*P];
   POSITION_T currQBitPos[M],currQBlkPos[M];  //  POSITION_T == uint32_t
   DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
   int check;
   int imax = ITERATIONS_MAX;
   unsigned int synd_corrt_vec[][2]= {SYNDROME_TRESH_LOOKUP_TABLE};

   do {
      gf2x_copy(currSyndrome, privateSyndrome);
      memset(unsatParityChecks,0x00,N0*P*sizeof(uint8_t));

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
          /* vectorized upc computation. Coalescing factor DIGIT_SIZE_B */
          int vectorIdx;
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

          /* Finalize computation  with trailing P % DIGIT_SIZE_B  UPCs*/
          int vecSyndBitsWordIdx;
          DIGIT vecSyndBits[(DV+DIGIT_SIZE_B-1)/DIGIT_SIZE_B] = {0};
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
          for(int vecElem=0; vecElem < P%DIGIT_SIZE_B; vecElem++) {
            uint8_t upcDigit = 0;
            for(vecSyndBitsWordIdx = 0; vecSyndBitsWordIdx < (DV+DIGIT_SIZE_B-1)/DIGIT_SIZE_B; vecSyndBitsWordIdx++){
                   DIGIT synVec = vecSyndBits[vecSyndBitsWordIdx];
                   upcDigit += __builtin_popcountll( (unsigned long long int) (synVec & 0x0101010101010101ULL));
                   vecSyndBits[vecSyndBitsWordIdx]= synVec >> 1;
            }
            unsatParityChecks[i*P+vectorIdx*DIGIT_SIZE_B+vecElem] = upcDigit;
          }
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
      for (int i = 0; i < N0; i++) {
         for (int j = 0; j < P; j++) {
            int currQoneIdx = 0; // position in the column of QtrPosOnes[][...]
            int endQblockIdx = 0;
            int correlation =0;

            for (int blockIdx = 0; blockIdx < N0; blockIdx++) {
               endQblockIdx += qBlockWeights[blockIdx][i];
               int currblockoffset = blockIdx*P;
               for (; currQoneIdx < endQblockIdx; currQoneIdx++) {
                  int tmp = QtrPosOnes[i][currQoneIdx]+j;
                  tmp = tmp >= P ? tmp - P : tmp;
                  currQBitPos[currQoneIdx] = tmp;
                  currQBlkPos[currQoneIdx] = blockIdx;
                  correlation += unsatParityChecks[tmp + currblockoffset];
               }
            }
            /* Correlation based flipping */
            if (correlation > corrt_syndrome_based) {
               gf2x_toggle_coeff(out+NUM_DIGITS_GF2X_ELEMENT*i, j);
               for (int v = 0; v < M; v++) {
                  unsigned syndromePosToFlip;
                  for (int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
                     syndromePosToFlip = (HtrPosOnes[currQBlkPos[v]][HtrOneIdx] + currQBitPos[v] );
                     syndromePosToFlip = syndromePosToFlip >= P ? syndromePosToFlip - P : syndromePosToFlip;
                     gf2x_toggle_coeff(privateSyndrome, syndromePosToFlip);
                  }
               } // end for v
            } // end if
         } // end for j
      } // end for i

      imax = imax - 1;
      check = 0;
      while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check++] == 0);

   } while (imax != 0 && check < NUM_DIGITS_GF2X_ELEMENT);

   return (check == NUM_DIGITS_GF2X_ELEMENT);
}  // end QdecodeSyndromeThresh_bitFlip_sparse
