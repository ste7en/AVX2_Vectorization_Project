/**
 *
 * <bf_decoding.c>
 *
 * @version 2.0 (March 2019)
 *
 * Reference ISO-C11 Implementation of the LEDAcrypt KEM cipher using GCC built-ins.
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
#include "architecture_detect.h"
#include <string.h>
#include <assert.h>
#include <immintrin.h>

#define ROTBYTE(a)   ( (a << 8) | (a >> (DIGIT_SIZE_b - 8)) )
#define ROTUPC(a)   ( (a >> 8) | (a << (DIGIT_SIZE_b - 8)) )
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

#ifdef HIGH_PERFORMANCE_X86_64

/******************** START of functions' definitions for vector operations *************************/
static inline __m256i permute_row(__m256i row){
    row = _mm256_shuffle_epi8(row, _mm256_set_epi8(15, 11, 7, 3,
                                                   14, 10, 6, 2,
                                                   13,  9, 5, 1,
                                                   12,  8, 4, 0,
                                                   15, 11, 7, 3,
                                                   14, 10, 6, 2,
                                                   13,  9, 5, 1,
                                                   12,  8, 4, 0));
    row = _mm256_permutevar8x32_epi32(row, _mm256_set_epi32(7,3,6,2,5,1,4,0));
    return row;
}

static inline void transpose_matrix_32_8(__m256i *restrict matrix){
    __m256  row0  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[0]));
    __m256  row1  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[1]));
    __m256  row2  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[2]));
    __m256  row3  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[3]));
    __m256  row4  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[4]));
    __m256  row5  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[5]));
    __m256  row6  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[6]));
    __m256  row7  = _mm256_castsi256_ps(_mm256_loadu_si256(&matrix[7]));
    __m256  __t0  = _mm256_unpacklo_ps(row0, row1);
    __m256  __t1  = _mm256_unpackhi_ps(row0, row1);
    __m256  __t2  = _mm256_unpacklo_ps(row2, row3);
    __m256  __t3  = _mm256_unpackhi_ps(row2, row3);
    __m256  __t4  = _mm256_unpacklo_ps(row4, row5);
    __m256  __t5  = _mm256_unpackhi_ps(row4, row5);
    __m256  __t6  = _mm256_unpacklo_ps(row6, row7);
    __m256  __t7  = _mm256_unpackhi_ps(row6, row7);
    __m256  __tt0 = _mm256_shuffle_ps(__t0,__t2, 0x44);
    __m256  __tt1 = _mm256_shuffle_ps(__t0,__t2, 0xEE);
    __m256  __tt2 = _mm256_shuffle_ps(__t1,__t3, 0x44);
    __m256  __tt3 = _mm256_shuffle_ps(__t1,__t3, 0xEE);
    __m256  __tt4 = _mm256_shuffle_ps(__t4,__t6, 0x44);
    __m256  __tt5 = _mm256_shuffle_ps(__t4,__t6, 0xEE);
    __m256  __tt6 = _mm256_shuffle_ps(__t5,__t7, 0x44);
    __m256  __tt7 = _mm256_shuffle_ps(__t5,__t7, 0xEE);
            row0  = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
            row1  = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
            row2  = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
            row3  = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
            row4  = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
            row5  = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
            row6  = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
            row7  = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
            /* End of 8x8 float transpose, cast the results back to _si256: */
    __m256i row0i = _mm256_castps_si256(row0);
    __m256i row1i = _mm256_castps_si256(row1);
    __m256i row2i = _mm256_castps_si256(row2);
    __m256i row3i = _mm256_castps_si256(row3);
    __m256i row4i = _mm256_castps_si256(row4);
    __m256i row5i = _mm256_castps_si256(row5);
    __m256i row6i = _mm256_castps_si256(row6);
    __m256i row7i = _mm256_castps_si256(row7);
            /* Now we only need a simple row permutation to get the right result: */
            row0i = permute_row(row0i);
            row1i = permute_row(row1i);
            row2i = permute_row(row2i);
            row3i = permute_row(row3i);
            row4i = permute_row(row4i);
            row5i = permute_row(row5i);
            row6i = permute_row(row6i);
            row7i = permute_row(row7i);
            _mm256_storeu_si256(&matrix[0], row0i);
            _mm256_storeu_si256(&matrix[1], row1i);
            _mm256_storeu_si256(&matrix[2], row2i);
            _mm256_storeu_si256(&matrix[3], row3i);
            _mm256_storeu_si256(&matrix[4], row4i);
            _mm256_storeu_si256(&matrix[5], row5i);
            _mm256_storeu_si256(&matrix[6], row6i);
            _mm256_storeu_si256(&matrix[7], row7i);
}

void print_matrix(__m256i* matrix, int rows, int cols){
    unsigned char* v;
    int i, j, k;
    /* Access matrix as chars */
    v = (unsigned char*)matrix;
    i = 0;
    /* Print the chars v[i] , i = 0, 1, 2, 3,..., 255                          */
    /* rows and cols only controls the positions of the new lines printf("\n") */
    for (k = 0; k < rows; k++){
        for (j = 0; j < cols; j++){
            printf("%4hhu", v[i]);
            i = i + 1;
        }
        printf("\n");
    }
    printf("\n");
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
#define SIZE_OF_UPC_VECTORIZED_READ 32
#define SIZE_OF_UPC_VECTORIZED_READ_AVX2 256
#define VECTYPE DIGIT
#define VECTYPE_SIZE_b DIGIT_SIZE_b
#define VECTYPE_SIZE_B DIGIT_SIZE_B
#define BIT_SIZE_UPC 8
#define VEC_COMB_MASK 0x0101010101010101ULL
#define AVX2_REG_SIZE_b 256
#define AVX2_REG_SIZE_B 32
   uint8_t unsatParityChecks[N0][ ROUND_UP(P,SIZE_OF_UPC_VECTORIZED_READ_AVX2)+SIZE_OF_UPC_VECTORIZED_READ_AVX2] = {{0}};
   POSITION_T currQBitPos[M];
   /* syndrome is endowed with cyclic padding in the leading word to avoid
    * boundary checks. The pad should be at least as long as the bit-len of one
    * vector of bits which is read during UPC computation */
   DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT+4];
   int check;
   int imax = ITERATIONS_MAX;
   unsigned int synd_corrt_vec[][2]= {SYNDROME_TRESH_LOOKUP_TABLE};
   // printf("%d\n", MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT);

   do {
      gf2x_copy(currSyndrome+4, privateSyndrome);
/*position of the first set bit in the word, counting 63 ... 0*/
#if (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT == (DIGIT_SIZE_b-1))
      memcpy(&currSyndrome[0], &currSyndrome[NUM_DIGITS_GF2X_ELEMENT], DIGIT_SIZE_B*4);
      //currSyndrome[0] = currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
#else
      currSyndrome[4] |= (currSyndrome[NUM_DIGITS_GF2X_ELEMENT+3] << (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1) );
      __m256i syndromePadding = _mm256_lddqu_si256((__m256i*)&currSyndrome[NUM_DIGITS_GF2X_ELEMENT]);
              syndromePadding = _mm256_SHIFT_RIGHT_bit(syndromePadding, (DIGIT_SIZE_b- (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1)));
      _mm256_storeu_si256( (__m256i *)&currSyndrome[0],syndromePadding);
      // currSyndrome[0] = currSyndrome[NUM_DIGITS_GF2X_ELEMENT] >> (DIGIT_SIZE_b- (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1));
#endif

// START BLOCK TO REMOVE - ONLY FOR TEST
   DIGIT currSyndrome_test[NUM_DIGITS_GF2X_ELEMENT+1];
   VECTYPE upcMat_test[BIT_SIZE_UPC];
   VECTYPE packedSynBits_test;
   int numDigitsGf2xElement = NUM_DIGITS_GF2X_ELEMENT;
   int roundedUp = ROUND_UP(P,SIZE_OF_UPC_VECTORIZED_READ)+SIZE_OF_UPC_VECTORIZED_READ;
   uint8_t unsatParityChecks_test[N0][ ROUND_UP(P,SIZE_OF_UPC_VECTORIZED_READ)+SIZE_OF_UPC_VECTORIZED_READ] = {{0}};
   gf2x_copy(currSyndrome_test+1, privateSyndrome);

      /*position of the first set bit in the word, counting 63 ... 0*/
   #if (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT == (DIGIT_SIZE_b-1))
      currSyndrome_test[0] = currSyndrome_test[NUM_DIGITS_GF2X_ELEMENT];
   #else
      currSyndrome_test[1] |= (currSyndrome_test[NUM_DIGITS_GF2X_ELEMENT] << (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1) );
      currSyndrome_test[0] = currSyndrome_test[NUM_DIGITS_GF2X_ELEMENT] >> (DIGIT_SIZE_b- (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1));
   #endif
// END BLOCK TO REMOVE - ONLY FOR TEST

      __m256i vecUpcMat[BIT_SIZE_UPC];
      __m256i packedSynBits = _mm256_setzero_si256();

      for (int i = 0; i < N0; i++) {
         // for(int valueIdx = 0, valueIdx_test = 0; valueIdx < P, valueIdx_test < P; valueIdx = valueIdx + AVX2_REG_SIZE_b, valueIdx_test = valueIdx_test + DIGIT_SIZE_b) {
         for(int valueIdx = 0; valueIdx < P; valueIdx = valueIdx + AVX2_REG_SIZE_b) {

            for (int upcMatRow = 0; upcMatRow < BIT_SIZE_UPC; upcMatRow++) {
               vecUpcMat[upcMatRow] = _mm256_setzero_si256();
            }
            // // START REMOVE
            // memset(upcMat_test, 0 , VECTYPE_SIZE_B*(VECTYPE_SIZE_b/BIT_SIZE_UPC));
            // // END REMOVE

            /* this fetches AVX2_REG_SIZE_b bits from each Htrpos, packed, and adds them to the 256 upc counters in upcmat */
            for(int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
               // TODO: - vectorize this modulo operation
               POSITION_T basePos = (HtrPosOnes[i][HtrOneIdx]+valueIdx);
               basePos = basePos % P ;

               // __m128i basePos = _mm_set1_epi32(tmp);
               /* lsb here is the one in base pos, others are subseq*/
               gf2x_get_M256_SIZE_coeff_vector_boundless(currSyndrome, basePos, &packedSynBits);

               //START REMOVE
               long long test[4] = {0};
               for (int valueIdx_test = valueIdx; valueIdx_test < valueIdx + 256; valueIdx_test = valueIdx_test + 64) {
                  POSITION_T basePos_test = (HtrPosOnes[i][HtrOneIdx]+valueIdx_test);
                  basePos_test = basePos_test % P ;
                  /* lsb here is the one in base pos, others are subseq*/
                  test[(valueIdx_test-valueIdx)/64] = gf2x_get_DIGIT_SIZE_coeff_vector_boundless(currSyndrome_test,basePos_test);
                  // packedSynBits_test = gf2x_get_DIGIT_SIZE_coeff_vector_boundless(currSyndrome_test,basePos_test);
               }
               printf("%d\n", memcmp(&test, &packedSynBits, 32));
               //END REMOVE

               for(int upcMatRow = 0; upcMatRow < BIT_SIZE_UPC; upcMatRow++) {
                  vecUpcMat[upcMatRow] = _mm256_add_epi64(vecUpcMat[upcMatRow],
                                                          _mm256_and_si256(packedSynBits,
                                                                           _mm256_set1_epi64x(VEC_COMB_MASK))
                                                         );
                  packedSynBits = _mm256_SHIFT_RIGHT_bit(packedSynBits, 0x01);
               }

               // //START REMOVE
               // for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
               //    upcMat_test[upcMatRow] += packedSynBits_test & VEC_COMB_MASK;
               //    packedSynBits_test = packedSynBits_test >> 1;
               // }/* end of for computing 64 upcs*/
               // //END REMOVE

            } /* end of for computing 256 upcs*/ // index HtrOneIdx

            /* commit computed UPCs in the upc vector, in the proper order.
             * UPCmat essentially needs transposition and linearization by row,
             * starting from the last row */
             transpose_matrix_32_8(vecUpcMat);

             for(int upcMatRow = 0; upcMatRow < BIT_SIZE_UPC; upcMatRow++) {
                // __m256i* vp = (__m256i *)(&unsatParityChecks[i][valueIdx+8*upcMatRow]);
                __m256i* vp = (__m256i *)(&unsatParityChecks[i][valueIdx+32*upcMatRow]);
                _mm256_storeu_si256(vp, vecUpcMat[upcMatRow]);
             }

   //           //START REMOVE
   //           /* commit computed UPCs in the upc vector, in the proper order.
   //           * upcMat_test essentially needs transposition and linearization by row,
   //           * starting from the last row */
   //           for(int upcMatCol = 0; upcMatCol < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatCol++){
   //                 VECTYPE upcBuf_test = 0;
   //                 for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
   //                 uint8_t matByte_test = upcMat_test[upcMatRow];
   // #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
   //                 upcBuf_test |=  ((VECTYPE)(matByte_test)) << (8*upcMatRow);
   // #else
   //                 upcBuf_test |=  (upcBuf_test << 8) + ((VECTYPE)(matByte_test));
   // #endif
   //                    upcMat_test[upcMatRow] = upcMat_test[upcMatRow] >> 8;
   //                 }
   //                 VECTYPE* vp_test = (VECTYPE *)(&unsatParityChecks_test[i][valueIdx_test+8*upcMatCol]);
   //                 *(vp_test) = upcBuf_test;
   //           }
   //           //END REMOVE

         } /* end of for valueIdx */
      } // end for i

// test

      for (int i = 0; i < N0; i++) {
        for (int valueIdx_test = 0; valueIdx_test < P; valueIdx_test = valueIdx_test + DIGIT_SIZE_b) {
           memset(upcMat_test, 0 , VECTYPE_SIZE_B*(VECTYPE_SIZE_b/BIT_SIZE_UPC));
           /* this fetches DIGIT_SIZE_b bits from each Htrpos, packed, and adds them to the 64 upc counters in upcMat_test */
           for(int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
                  POSITION_T basePos_test = (HtrPosOnes[i][HtrOneIdx]+valueIdx_test);
                  basePos_test = basePos_test %P ;
                  /* lsb here is the one in base pos, others are subseq*/
                  packedSynBits_test = gf2x_get_DIGIT_SIZE_coeff_vector_boundless(currSyndrome_test,basePos_test);
                  for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
                     upcMat_test[upcMatRow] += packedSynBits_test & VEC_COMB_MASK;
                     packedSynBits_test = packedSynBits_test >> 1;
                  }

          }/* end of for computing 64 upcs*/
           /* commit computed UPCs in the upc vector, in the proper order.
           * upcMat_test essentially needs transposition and linearization by row,
           * starting from the last row */
           for(int upcMatCol = 0; upcMatCol < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatCol++){
                VECTYPE upcBuf_test = 0;
                for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
                 uint8_t matByte_test = upcMat_test[upcMatRow];
 #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                 upcBuf_test |=  ((VECTYPE)(matByte_test)) << (8*upcMatRow);
 #else
                 upcBuf_test |=  (upcBuf_test << 8) + ((VECTYPE)(matByte_test));
 #endif
                    upcMat_test[upcMatRow] = upcMat_test[upcMatRow] >> 8;
                }
                VECTYPE* vp_test = (VECTYPE *)(&unsatParityChecks_test[i][valueIdx_test+8*upcMatCol]);
                *(vp_test) = upcBuf_test;
           } // end for upcMatCol
        } // end for valueIdx_test
      } // end for i

printf("\nVect:\n");
print_matrix(unsatParityChecks, 8, 32);
printf("\nNon vect:\n");
print_matrix(unsatParityChecks_test, 8, 32);
// end test

      /* circular padding of unsatisfiedParityChecks so that the vector
       * correlation computation does not need to wraparound loads*/
      for (int i = 0; i < N0; i++) {
          for(int j = 0; j < SIZE_OF_UPC_VECTORIZED_READ_AVX2; j++)
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
         for (int j = 0; j <= P; j+=SIZE_OF_UPC_VECTORIZED_READ) {
            int currQoneIdx; // position in the column of QtrPosOnes[][...]

            /* vector of correlations computed in a single sweep of AVX2 vectorized
             * computation.  Stored with type punning as a 4 item uint64_t
             * vectorized corrs. */
            uint64_t correlation = 0;
            uint64_t correlation_avx[4] = {0};
            __m256i vecCorr = _mm256_setzero_si256();
            __m256i vecUpcToAdd;
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
                tmp = tmp % P ;  /* this will need to be checked before and have a split vector load */
                currQBitPos[currQoneIdx] = tmp;  /* only a single base index must be kept per vector load*/
                vecUpcToAdd = _mm256_lddqu_si256((__m256i *) (&unsatParityChecks[qBlockOffsets[currQoneIdx]][tmp]));
                vecCorr = _mm256_add_epi8(vecUpcToAdd,vecCorr);
            }
            _mm256_storeu_si256( (__m256i *) correlation_avx,vecCorr);


            for(int vecCorrDeltaJ = 0 ; vecCorrDeltaJ < SIZE_OF_UPC_VECTORIZED_READ/DIGIT_SIZE_B; vecCorrDeltaJ++) {
            /* Correlation based flipping */
            correlation = correlation_avx[vecCorrDeltaJ];
               for (int delta_j = 0 ;
                    delta_j < DIGIT_SIZE_B && (j+vecCorrDeltaJ*DIGIT_SIZE_B+delta_j < P) ;
                    delta_j++){
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                  uint64_t single_correl = ( correlation >> (8*delta_j) ) & 0xFF;
#else
                  uint64_t single_correl = ( correlation >> (8*(DIGIT_SIZE_B-1-delta_j)) ) & 0xFF;
#endif
                  if ((single_correl > corrt_syndrome_based)) { /* correlation will always be null for garbage bits */
                     gf2x_toggle_coeff(out+NUM_DIGITS_GF2X_ELEMENT*i, (j+vecCorrDeltaJ*DIGIT_SIZE_B+delta_j)%P); /*this can be vectorized if a vector of correlations are available*/
                     for (int v = 0; v < M; v++) { /* the same goes here */
                        unsigned syndromePosToFlip;
                        for (int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
                           syndromePosToFlip = (HtrPosOnes[qBlockIdxes[v]][HtrOneIdx] +
                                                (currQBitPos[v]+vecCorrDeltaJ*DIGIT_SIZE_B+delta_j)%P );
                           syndromePosToFlip = syndromePosToFlip >= P ? syndromePosToFlip - P : syndromePosToFlip;
                           gf2x_toggle_coeff(privateSyndrome, syndromePosToFlip);
                        }
                     } // end for v
                  } // end if
               } // end for flipping correlations exceeding threshold
            } // end of for selecting part of correlation vector to analyze
         } // end for j
      } // end for i

      imax = imax - 1;
      check = 0;
      while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check] == 0) {
          check++;
      }

   } while (imax != 0 && check < NUM_DIGITS_GF2X_ELEMENT);

   return (check == NUM_DIGITS_GF2X_ELEMENT);
}  // end QdecodeSyndromeThresh_bitFlip_sparse

/*C fallback if there are no AVX2*/
#else

// int bf_decoding(DIGIT out[], // N0 polynomials
//                 const POSITION_T HtrPosOnes[N0][DV],
//                 const POSITION_T QtrPosOnes[N0][M],
//                 DIGIT privateSyndrome[]  //  1 polynomial
//                )
// {
// #if P < 64
// #error The circulant block size should exceed 64
// #endif
//
//    /* The unsatisfied parity checks are kept as N0 vectors , each one having
//     * a length rounded up to the closest multiple of the coalescing factor
//     * DIGIT_SIZE_B, plus one entire digit, and with the last digit padded with zeroes.
//     * This strategy allows to replicate an entire digit at the end of the computed UPCs
//     * effectively removing altogether the (key-dependent) checks which would happen
//     * in the correlation computation process */
// #define SIZE_OF_UPC_VECTORIZED_READ 32
//    uint8_t unsatParityChecks[N0][ ROUND_UP(P,SIZE_OF_UPC_VECTORIZED_READ)+SIZE_OF_UPC_VECTORIZED_READ] = {{0}};
//    POSITION_T currQBitPos[M];
//    /* syndrome is endowed with cyclic padding in the leading word to avoid
//     * boundary checks. The pad should be at least as long as the bit-len of one
//     * vector of bits which is read during UPC computation */
//    DIGIT currSyndrome[NUM_DIGITS_GF2X_ELEMENT+1];
//    int check;
//    int imax = ITERATIONS_MAX;
//    unsigned int synd_corrt_vec[][2]= {SYNDROME_TRESH_LOOKUP_TABLE};
//
//    do {
//       gf2x_copy(currSyndrome+1, privateSyndrome);
//
// /*position of the first set bit in the word, counting 63 ... 0*/
// #if (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT == (DIGIT_SIZE_b-1))
//       currSyndrome[0] = currSyndrome[NUM_DIGITS_GF2X_ELEMENT];
// #else
//        currSyndrome[1] |= (currSyndrome[NUM_DIGITS_GF2X_ELEMENT] << (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1) );
//        currSyndrome[0] = currSyndrome[NUM_DIGITS_GF2X_ELEMENT] >> (DIGIT_SIZE_b- (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT+1));
// #endif
//
// #define VECTYPE DIGIT
// #define VECTYPE_SIZE_b DIGIT_SIZE_b
// #define VECTYPE_SIZE_B DIGIT_SIZE_B
// #define BIT_SIZE_UPC 8
// #define VEC_COMB_MASK 0x0101010101010101ULL
//
//     VECTYPE upcMat[VECTYPE_SIZE_b/BIT_SIZE_UPC];
//     VECTYPE packedSynBits;
//     for (int i = 0; i < N0; i++) {
//        for (int valueIdx = 0; valueIdx < P; valueIdx = valueIdx + DIGIT_SIZE_b) { /* this index will be vectorized*/
//           memset(upcMat, 0 , VECTYPE_SIZE_B*(VECTYPE_SIZE_b/BIT_SIZE_UPC));
//           /* this fetches DIGIT_SIZE_b bits from each Htrpos, packed, and adds them to the 64 upc counters in upcmat */
//           for(int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
//                 POSITION_T basePos = (HtrPosOnes[i][HtrOneIdx]+valueIdx);
//                 basePos = basePos %P ;
//                 /* lsb here is the one in base pos, others are subseq*/
//                 packedSynBits = gf2x_get_DIGIT_SIZE_coeff_vector_boundless(currSyndrome,basePos);
//                 for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
//                     upcMat[upcMatRow] += packedSynBits & VEC_COMB_MASK;
//                     packedSynBits = packedSynBits >> 1;
//                 }
//
//               }/* end of for computing 64 upcs*/
//          /* commit computed UPCs in the upc vector, in the proper order.
//           * UPCmat essentially needs transposition and linearization by row,
//           * starting from the last row */
//           for(int upcMatCol = 0; upcMatCol < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatCol++){
//               VECTYPE upcBuf = 0;
//               for(int upcMatRow = 0; upcMatRow < VECTYPE_SIZE_b/BIT_SIZE_UPC; upcMatRow++){
//                uint8_t matByte = upcMat[upcMatRow];
// #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
//                upcBuf |=  ((VECTYPE)(matByte)) << (8*upcMatRow);
// #else
//                upcBuf |=  (upcBuf << 8) + ((VECTYPE)(matByte));
// #endif
//                   upcMat[upcMatRow] = upcMat[upcMatRow] >> 8;
//               }
//               VECTYPE* vp = (VECTYPE *)(&unsatParityChecks[i][valueIdx+8*upcMatCol]);
//               *(vp) = upcBuf;
//           }
//          }
//      }
//
//       /* padding unsatisfiedParityChecks */
//       for (int i = 0; i < N0; i++) {
//           for(int j = 0; j < DIGIT_SIZE_B; j++)
//            unsatParityChecks[i][P+j] = unsatParityChecks[i][j];
//       }
//
//       // computation of syndrome weight and threshold determination
//       int syndrome_wt = population_count(currSyndrome);
//       int min_idx=0;
//       int max_idx;
//       max_idx = sizeof(synd_corrt_vec)/(2*sizeof(unsigned int)) - 1;
//       int thresh_table_idx = (min_idx + max_idx)/2;
//       while(min_idx< max_idx) {
//          if (synd_corrt_vec[thresh_table_idx][0] <= syndrome_wt) {
//             min_idx = thresh_table_idx +1;
//          } else {
//             max_idx = thresh_table_idx -1;
//          }
//          thresh_table_idx = (min_idx +max_idx)/2;
//       }
//       int corrt_syndrome_based=synd_corrt_vec[thresh_table_idx][1];
//
//       //Computation of correlation  with a full Q matrix
//       for (int i = 0; i < N0; i++) {
//             uint8_t qBlockIdxes[M];
//             uint16_t qBlockOffsets[M];
//             uint8_t qBlockIdxesCursor = 0, linearized_cursor = 0;
//             int endQblockIdx = 0;
//             for (int blockIdx = 0; blockIdx < N0; blockIdx++) {
//               endQblockIdx += qBlockWeights[blockIdx][i];
//               for (; linearized_cursor < endQblockIdx; linearized_cursor++) {
//                 qBlockIdxes[linearized_cursor] = qBlockIdxesCursor;
//                 qBlockOffsets[linearized_cursor] =  (uint16_t) qBlockIdxesCursor;
//               }
//               qBlockIdxesCursor++;
//             }
//          for (int j = 0; j <= P; j+=DIGIT_SIZE_B) {
//             int currQoneIdx; // position in the column of QtrPosOnes[][...]
//             /* a vector of (single byte wide) correlation values is computed
//              * at each iteration of the outer loop. 8x vectorization is possible
//              * with uint64_t. The most significant byte in the uint64_t vector
//              * of uint8_t is the one matching the outer loop index position*/
//             uint64_t correlation = 0;
//
//             /* a single correlation value is the sum of m "block circulant positioned" values of UPC */
//             /* I need a 8x storage of UPCs which I can load as 8x bytes from upc vector starting from
//              * the block circulant position.
//              * A trivial implementation will thus have m variables containing 8 consecutive UPCs.
//              *
//              * A more reasonable one uses a single variable which gets filled
//              * with 8x UPCs per shot, bar circulant cases related to 8 consecutive correlation
//              * computations, and a single circulant "one" from Q (ideal opts will unroll).
//              * Since the outer loop (j indexed, at line 183) is vectorized, a trailer of the last P%DIGIT_SIZE_b should be added
//              * (or not: a single vector computation discarding part of the results in correlation is equally fine) */
//
//             /* This vectorization strategy will benefit from having a vector with the block circulant pos.s in Q
//              * and another one with the block idxes. Both can be vector packed: pos.s are 16 bit wide, block idx.es are 2b wide
//              * FTTB I will generate both of them at runtime, but the latter can be generated as a #define.*/
//
//             for (currQoneIdx = 0; currQoneIdx < M; currQoneIdx++) {
//                 int tmp = QtrPosOnes[i][currQoneIdx]+j; /* this must turn into a vector load of 8x UPCs with consecutive j indices */
//                 tmp = tmp % P ;  /* this will need to be checked before and have a split vector load */
//                 currQBitPos[currQoneIdx] = tmp;  /* only a single base index must be kept per vector load*/
//
//                 uint64_t * vp = (uint64_t*)(&unsatParityChecks[qBlockOffsets[currQoneIdx]][tmp]);
//                 correlation += *vp;
//             }
//
//             /* Correlation based flipping */
//             for (int delta_j = 0 ; delta_j < DIGIT_SIZE_B && (j+delta_j < P) ; delta_j++){
// #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
//                uint64_t single_correl = ( correlation >> (8*delta_j) ) & 0xFF;
// #else
//                uint64_t single_correl = ( correlation >> (8*(DIGIT_SIZE_B-1-delta_j)) ) & 0xFF;
// #endif
//                if ((single_correl > corrt_syndrome_based)) { /* correlation will always be null for garbage bits */
//                   gf2x_toggle_coeff(out+NUM_DIGITS_GF2X_ELEMENT*i, (j+delta_j)%P); /*this can be vectorized if a vector of correlations are available*/
//                   for (int v = 0; v < M; v++) { /* the same goes here */
//                      unsigned syndromePosToFlip;
//                      for (int HtrOneIdx = 0; HtrOneIdx < DV; HtrOneIdx++) {
//                         syndromePosToFlip = (HtrPosOnes[qBlockIdxes[v]][HtrOneIdx] + (currQBitPos[v]+delta_j)%P );
//                         syndromePosToFlip = syndromePosToFlip >= P ? syndromePosToFlip - P : syndromePosToFlip;
//                         gf2x_toggle_coeff(privateSyndrome, syndromePosToFlip);
//                      }
//                   } // end for v
//                } // end if
//             } // end for flipping correlations exceeding threshold
//          } // end for j
//       } // end for i
//
//       imax = imax - 1;
//       check = 0;
//       while (check < NUM_DIGITS_GF2X_ELEMENT && privateSyndrome[check] == 0) {
//           check++;
//       }
//
//    } while (imax != 0 && check < NUM_DIGITS_GF2X_ELEMENT);
//
//    return (check == NUM_DIGITS_GF2X_ELEMENT);
// }  // end QdecodeSyndromeThresh_bitFlip_sparse
#endif
