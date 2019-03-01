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
#define ROUND_UP(amount, round_amt) ( ((amount+round_amt-1)/round_amt)*round_amt )

#if (DIGIT_MAX == UINT64_MAX)
#define DIGIT_SIZE_b_EXPONENT 6
#elif (DIGIT_MAX == UINT32_MAX)
#define DIGIT_SIZE_b_EXPONENT 5
#elif (DIGIT_MAX == UINT16_MAX)
#define DIGIT_SIZE_b_EXPONENT 4
#elif (DIGIT_MAX == UINT8_MAX)
#define DIGIT_SIZE_b_EXPONENT 3
#else
#error "unable to find the bitsize of size_t"
#endif
/******************** START of functions' definitions for vector operations *************************/

static inline __m128i _mm256_extractf128i_lower(__m256i a) {
   return _mm256_extractf128_si256(a, 0x00);
}

static inline __m128i _mm256_extractf128i_upper(__m256i a) {
   return _mm256_extractf128_si256(a, 0x01);
}

static inline void get_64_coeff_vector(const DIGIT poly[],
                                       const __m256i first_exponent_vector,
                                       const __m256i *restrict __lowerResult,
                                       const __m256i *restrict __upperResult
                                       )
{

#ifdef HIGH_PERFORMANCE_X86_64
   __m256i addend       = _mm256_set1_epi32(NUM_DIGITS_GF2X_ELEMENT*DIGIT_SIZE_b-1);
   __m256i straightIdx  = _mm256_sub_epi32 (addend, first_exponent_vector);
   // division by a power of two becomes a logic right shift
   __m256i digitIdx     = _mm256_srli_epi32(straightIdx, DIGIT_SIZE_b_EXPONENT);

   // Gather operation to load 8x 64-bit digits in two registers
   __m128i lowerDigitIdx   = _mm256_extractf128i_lower (digitIdx);
   __m256i lowerLsw        = _mm256_i32gather_epi64 (poly, lowerDigitIdx, 1);
   __m128i upperDigitIdx   = _mm256_extractf128i_upper (digitIdx);
   __m256i upperLsw        = _mm256_i32gather_epi64 (poly, upperDigitIdx, 1);

   /*the most significant valid bit in the lsw is always the 63rd, save for the case
    * where digitIdx is 0. In that case the slack bits are cut off, and the extraction
    * mask for the first word should be adjusted accordingly */
   __m256i msbPosInMSB  = _mm256_set1_epi32 (MSb_POSITION_IN_MSB_DIGIT_OF_ELEMENT); // digitIdx == 0
   __m256i msValidBit   = _mm256_set1_epi32 (DIGIT_SIZE_b-1); // digitIdx != 0
   // I'm using digitIdx as mask
   __m256i ceiling_pos  = _mm256_castps_si256( _mm256_blendv_ps (
                                                _mm256_castsi256_ps(msbPosInMSB),
                                                _mm256_castsi256_ps(msValidBit),
                                                _mm256_castsi256_ps(digitIdx)
                                                )
                                             );

   /* load word with wraparound */
   __m256i numDigitsElement   = _mm256_set1_epi32 (NUM_DIGITS_GF2X_ELEMENT-1);
   __m256i one_epi32          = _mm256_set1_epi32 (1);
   __m256i subtraction        = _mm256_sub_epi32  (digitIdx, one_epi32);
   /* Following a conditional assignment:
   *  digitIdx  = (digitIdx == 0) ? NUM_DIGITS_GF2X_ELEMENT-1 : digitIdx-1
   */
   digitIdx = _mm256_castps_si256( _mm256_blendv_ps (
                                    _mm256_castsi256_ps(numDigitsElement),
                                    _mm256_castsi256_ps(subtraction),
                                    _mm256_castsi256_ps(digitIdx)
                                    )
                                 );

   // Gather operation to load 8x 64-bit digits in two registers
   lowerDigitIdx     = _mm256_extractf128i_lower (digitIdx);
   __m256i lowerMsw  = _mm256_i32gather_epi64 (poly, lowerDigitIdx, 1);
   upperDigitIdx     = _mm256_extractf128i_upper (digitIdx);
   __m256i upperMsw  = _mm256_i32gather_epi64 (poly, upperDigitIdx, 1);

   // semantically equivalent to inDigitIdx = DIGIT_SIZE_b-1-(straightIdx % DIGIT_SIZE_b)
   __m256i digitBits = _mm256_set1_epi32(DIGIT_SIZE_b-1);
   __m256i modulo    = _mm256_and_si256 (straightIdx, digitBits); // n % 2^i = n & (2^i - 1)
   __m256i inDigitIdx= _mm256_abs_epi32 (
                           _mm256_sub_epi32 (
                              _mm256_sub_epi32(digitBits, one_epi32),
                              modulo
                           )
                        );

   // inDigitIdx is composed of 32-bit integers,
   // but I need 64-bit integers to compute the result
   __m128i lowerInDigitIdx = _mm256_extractf128i_lower(inDigitIdx);
   __m128i upperInDigitIdx = _mm256_extractf128i_upper(inDigitIdx);

   __m256i lowerInDigitIdx_epi64 = _mm256_cvtepu32_epi64(lowerInDigitIdx);
   __m256i upperInDigitIdx_epi64 = _mm256_cvtepu32_epi64(upperInDigitIdx);

#if P%DIGIT_SIZE_b < 32
   /* This case is managed to allow experimentation with parameter sets different
    * from the proposed ones.
    * It will yield a structural hindrance to constant time implementations*/

   // mask for the conditional assignment of result
   // if (digitIdx == 0 && 8 - (int)(DIGIT_SIZE_b-inDigitIdx) > (int)(P%DIGIT_SIZE_b))
   __m256i lowerDigitIdx_epi64   = _mm256_cvtepu32_epi64(lowerDigitIdx);
   __m256i upperDigitIdx_epi64   = _mm256_cvtepu32_epi64(upperDigitIdx);

   __m256i lowerRightAndOperand  = _mm256_cmpgt_epi64 (
                                    _mm256_sub_epi64 (
                                       _mm256_set1_epi64x(8-DIGIT_SIZE_b),
                                       lowerInDigitIdx_epi64
                                     ),
                                    _mm256_set1_epi64x(P%DIGIT_SIZE_b)
                                   );
   __m256i upperRightAndOperand  = _mm256_cmpgt_epi64 (
                                    _mm256_sub_epi64 (
                                       _mm256_set1_epi64x(8-DIGIT_SIZE_b),
                                       upperInDigitIdx_epi64
                                     ),
                                    _mm256_set1_epi64x(P%DIGIT_SIZE_b)
                                   );

  __m256i lowerConditionalResultMask = _mm256_and_si256 (
                                          _mm256_cmpeq_epi64 (
                                             lowerDigitIdx_epi64,
                                             _mm256_setzero_si256()
                                           ),
                                          lowerRightAndOperand
                                       );
  __m256i upperConditionalResultMask = _mm256_and_si256 (
                                          _mm256_cmpeq_epi64 (
                                             upperDigitIdx_epi64,
                                             _mm256_setzero_si256()
                                           ),
                                          upperRightAndOperand
                                       );
   /*
    *    Start of the logic true block of the if statement
    */

   // broadcasted value used to compute the conditional result
   __m256i bottomw = _mm256_set1_epi64x (poly[NUM_DIGITS_GF2X_ELEMENT-1]);

   // topmostBits = 8 - (P%DIGIT_SIZE_b) - (DIGIT_SIZE_b-inDigitIdx)
   __m256i lowerTopmostBits   = _mm256_sub_epi64 (
                                    _mm256_set1_epi64x (8 - (P%DIGIT_SIZE_b) - DIGIT_SIZE_b),
                                    lowerInDigitIdx_epi64
                                 );
   __m256i upperTopmostBits   = _mm256_sub_epi64 (
                                    _mm256_set1_epi64x (8 - (P%DIGIT_SIZE_b) - DIGIT_SIZE_b),
                                    upperInDigitIdx_epi64
                                 );
   // result = bottomw & ( (((DIGIT) 1) << topmostBits) -1)
   __m256i one_epi64          = _mm256_set1_epi64x (1);

   __m256i lowerIntermediateResultIfTrue = _mm256_and_si256 (
                                       bottomw,
                                       _mm256_sub_epi64 (
                                          _mm256_sll_epi64 (
                                             one_epi64,
                                             lowerTopmostBits
                                          ),
                                          one_epi64
                                       )
                                    );
   __m256i upperIntermediateResultIfTrue = _mm256_and_si256 (
                                       bottomw,
                                       _mm256_sub_epi64 (
                                          _mm256_sll_epi64 (
                                             one_epi64,
                                             upperTopmostBits
                                          ),
                                          one_epi64
                                       )
                                    );
   // result = result << (P%DIGIT_SIZE_b)
   lowerIntermediateResultIfTrue = _mm256_slli_epi64 (lowerIntermediateResultIfTrue, P%DIGIT_SIZE_b);
   upperIntermediateResultIfTrue = _mm256_slli_epi64 (upperIntermediateResultIfTrue, P%DIGIT_SIZE_b);

   // result |=  poly[0]
   __m256i orOperand       = _mm256_set1_epi64x(poly[0]);

   lowerIntermediateResultIfTrue = _mm256_or_si256 (lowerIntermediateResultIfTrue, orOperand);
   upperIntermediateResultIfTrue = _mm256_or_si256 (upperIntermediateResultIfTrue, orOperand);

   // result = result << (8-topmostBits-(P%DIGIT_SIZE_b))
   __m256i lowerShiftOperand  = _mm256_sub_epi64(
                                 _mm256_set1_epi64x(8-(P%DIGIT_SIZE_b)),
                                 lowerTopmostBits
                                );
   __m256i upperShiftOperand  = _mm256_sub_epi64(
                                 _mm256_set1_epi64x(8-(P%DIGIT_SIZE_b)),
                                 upperTopmostBits
                                );
   lowerIntermediateResultIfTrue    = _mm256_sllv_epi64 (lowerIntermediateResultIfTrue, lowerShiftOperand);
   upperIntermediateResultIfTrue    = _mm256_sllv_epi64 (upperIntermediateResultIfTrue, upperShiftOperand);

   // DIGIT vectorExtractionMask = (1 << (8-topmostBits-(P%DIGIT_SIZE_b)))-1
   __m256i lowerVectorExtractionMask = _mm256_sub_epi64(
                                          _mm256_sllv_epi64(
                                             one_epi64,
                                             lowerShiftOperand
                                          ),
                                          one_epi64
                                       );
   __m256i upperVectorExtractionMask = _mm256_sub_epi64(
                                          _mm256_sllv_epi64(
                                             one_epi64,
                                             upperShiftOperand
                                          ),
                                          one_epi64
                                       );

   // vectorExtractionMask = vectorExtractionMask << (DIGIT_SIZE_b - (8-topmostBits-(P%DIGIT_SIZE_b)))
   __m256i digitSizeBit = _mm256_set1_epi64x(DIGIT_SIZE_b);
   lowerVectorExtractionMask = _mm256_sllv_epi64(
                                 lowerVectorExtractionMask,
                                 _mm256_sub_epi64(
                                    digitSizeBit,
                                    lowerShiftOperand
                                 )
                               );
   upperVectorExtractionMask = _mm256_sllv_epi64(
                                 upperVectorExtractionMask,
                                 _mm256_sub_epi64(
                                    digitSizeBit,
                                    upperShiftOperand
                                 )
                               );

   // result |= ( (poly[1]  & vectorExtractionMask) >> (DIGIT_SIZE_b - (8-topmostBits-(P%DIGIT_SIZE_b))) )
   lowerIntermediateResultIfTrue =  _mm256_or_si256(
                                 _mm256_srlv_epi64 (
                                    _mm256_and_si256 (
                                       _mm256_set1_epi64x(poly[1]),
                                       lowerVectorExtractionMask
                                    ),
                                    _mm256_sub_epi64(
                                       digitSizeBit,
                                       lowerShiftOperand
                                    )
                                 ),
                                 lowerIntermediateResultIfTrue
                              );
   upperIntermediateResultIfTrue = _mm256_or_si256(
                                 _mm256_srlv_epi64 (
                                    _mm256_and_si256 (
                                       _mm256_set1_epi64x(poly[1]),
                                       upperVectorExtractionMask
                                    ),
                                    _mm256_sub_epi64(
                                       digitSizeBit,
                                       upperShiftOperand
                                    )
                                 ),
                                 upperIntermediateResultIfTrue
                              );
   /*
    *    Start of the logic false block of the if statement (else)
    */

   // int excessMSb = inDigitIdx + 7 - ceiling_pos
   __m256i excessMSb       = _mm256_sub_epi32(
                              _mm256_add_epi32(
                                 inDigitIdx,
                                 _mm256_set1_epi32(7)
                                 ),
                              ceiling_pos
                             );
   // excessMSb = excessMSb < 0 ? 0 : excessMSb
   __m256i zero            = _mm256_setzero_si256 ();
   __m256i conditionalMask = _mm256_cmpgt_epi32 (zero, excessMSb);
   excessMSb               = _mm256_castps_si256 ( _mm256_blendv_ps (
                                                      _mm256_castsi256_ps(excessMSb),
                                                      _mm256_castsi256_ps(zero),
                                                      _mm256_castsi256_ps(conditionalMask)
                                                   )
                                                 );

   // result = msw & ( (((DIGIT) 1) << excessMSb) -1)
   __m256i lowerExcessMSb_epi64 = _mm256_cvtepu32_epi64 ( _mm256_extractf128i_lower(excessMSb) );
   __m256i upperExcessMSb_epi64 = _mm256_cvtepu32_epi64 ( _mm256_extractf128i_upper(excessMSb) );
   __m256i lowerIntermediateResultIfFalse = _mm256_and_si256 (
                                             lowerMsw,
                                             _mm256_sub_epi64 (
                                                _mm256_sllv_epi64 (
                                                   one_epi64,
                                                   lowerExcessMSb_epi64),
                                                one_epi64
                                             )
                                            );
   __m256i upperIntermediateResultIfFalse = _mm256_and_si256 (
                                             upperMsw,
                                             _mm256_sub_epi64 (
                                                _mm256_sllv_epi64 (
                                                   one_epi64,
                                                   upperExcessMSb_epi64),
                                                one_epi64
                                             )
                                            );

   // result = result << (8-excessMSb)
   lowerIntermediateResultIfFalse = _mm256_sllv_epi64 (
                                       lowerIntermediateResultIfFalse,
                                       _mm256_sub_epi64 (
                                          _mm256_set1_epi64x(8),
                                          lowerExcessMSb_epi64 )
                                    );
   upperIntermediateResultIfFalse = _mm256_sllv_epi64 (
                                       upperIntermediateResultIfFalse,
                                       _mm256_sub_epi64 (
                                          _mm256_set1_epi64x(8),
                                          upperExcessMSb_epi64 )
                                    );

   /*no specialization needed as the slack bits are kept clear */
   // DIGIT vectorExtractionMask = (1 << 8) -1;
   vectorExtractionMask = _mm256_sub_epi64 (
                           _mm256_sllv_epi64 (
                              one_epi64,
                              _mm256_set1_epi64x(8)
                           ),
                           _mm256_set1_epi64x(1)
                          );

   // result |= (lsw & (vectorExtractionMask << inDigitIdx)) >> inDigitIdx;
   lowerIntermediateResultIfFalse = _mm256_or_si256 (
                                       lowerIntermediateResultIfFalse,
                                       _mm256_srlv_epi64 (
                                          _mm256_and_si256 (
                                             lowerLsw,
                                             _mm256_sllv_epi64(
                                                vectorExtractionMask,
                                                lowerInDigitIdx_epi64
                                             ) // end shift left
                                          ),
                                          lowerInDigitIdx_epi64
                                       ) // end shift right
                                    );
   upperIntermediateResultIfFalse = _mm256_or_si256 (
                                       upperIntermediateResultIfFalse,
                                       _mm256_srlv_epi64 (
                                          _mm256_and_si256 (
                                             upperLsw,
                                             _mm256_sllv_epi64(
                                                vectorExtractionMask,
                                                upperInDigitIdx_epi64
                                             ) // end shift left
                                          ),
                                          upperInDigitIdx_epi64
                                       ) // end shift right
                                    );

   // Conditional assignment of lower and upper result

   __m256i lowerResult = _mm256_castpd_si256 ( _mm256_blendv_pd (
                                                _mm256_castsi256_pd (lowerIntermediateResultIfFalse),
                                                _mm256_castsi256_pd (lowerIntermediateResultIfTrue),
                                                _mm256_castsi256_pd (lowerConditionalResultMask)
                                               )
                                           );
   __m256i upperResult = _mm256_castpd_si256 ( _mm256_blendv_pd (
                                                _mm256_castsi256_pd (upperIntermediateResultIfFalse),
                                                _mm256_castsi256_pd (upperIntermediateResultIfTrue),
                                                _mm256_castsi256_pd (upperConditionalResultMask)
                                               )
                                           );
#else // P%DIGIT_SIZE_b >= 8
   __m256i lowerResult;
   __m256i upperResult;

   /* one-byte wide mask to perform extraction */
   // int excessMSb = inDigitIdx + 7 - ceiling_pos
   __m256i excessMSb = _mm256_sub_epi32 (
                        _mm256_add_epi32 (
                           inDigitIdx,
                           _mm256_set1_epi32 (7)
                           ),
                        ceiling_pos);
   // excessMSb = excessMSb < 0 ? 0 : excessMSb
   __m256i conditionalMask = _mm256_cmpgt_epi32 (_mm256_setzero_si256, excessMSb);
   excessMSb = _mm256_blendv_ps(
                  _mm256_castsi256_ps(_mm256_setzero_si256),
                  _mm256_castsi256_ps(excessMSb),
                  _mm256_castsi256_ps(conditionalMask)
               );

   // result = msw & ( (((DIGIT) 1) << excessMSb) -1)
   __m256i one_epi64             = _mm256_set1_epi64x (1);
   __m256i lowerExcessMSb_epi64  = _mm256_cvtepu32_epi64 ( _mm256_extractf128i_lower(excessMSb) );
   __m256i upperExcessMSb_epi64  = _mm256_cvtepu32_epi64 ( _mm256_extractf128i_upper(excessMSb) );

   __m256i lowerResult           = _mm256_and_si256 (
                                    lowerMsw,
                                    _mm256_sub_epi64 (
                                       _mm256_sllv_epi64 (
                                          one_epi64,
                                          lowerExcessMSb_epi64),
                                       one_epi64
                                    )
                                   );
   __m256i upperResult           = _mm256_and_si256 (
                                    upperMsw,
                                    _mm256_sub_epi64 (
                                       _mm256_sllv_epi64 (
                                          one_epi64,
                                          upperExcessMSb_epi64),
                                       one_epi64
                                    )
                                   );

   // result = result << (8-excessMSb)
   lowerResult = _mm256_sllv_epi64 (
                  lowerIntermediateResultIfFalse,
                  _mm256_sub_epi64 (
                     _mm256_set1_epi64x(8),
                     lowerExcessMSb_epi64 )
                 );
   upperResult = _mm256_sllv_epi64 (
                  upperIntermediateResultIfFalse,
                  _mm256_sub_epi64 (
                     _mm256_set1_epi64x(8),
                     upperExcessMSb_epi64 )
                 );

   /*no specialization needed as the slack bits are kept clear */
   // DIGIT vectorExtractionMask = (1 << 8) -1;
   __m256i vectorExtractionMask = _mm256_sub_epi64 (
                                    _mm256_sllv_epi64 (
                                       one_epi64,
                                       _mm256_set1_epi64x(8)
                                    ),
                                    _mm256_set1_epi64x(1)
                                  );

   // result |= (lsw & (vectorExtractionMask << inDigitIdx)) >> inDigitIdx;
   lowerResult = _mm256_or_si256 (
                   lowerResult,
                   _mm256_srlv_epi64 (
                      _mm256_and_si256 (
                         lowerLsw,
                         _mm256_sllv_epi64(
                            vectorExtractionMask,
                            lowerInDigitIdx_epi64
                         ) // end shift left
                      ),
                      lowerInDigitIdx_epi64
                   ) // end shift right
                 );
   upperResult = _mm256_or_si256 (
                   upperResult,
                   _mm256_srlv_epi64 (
                      _mm256_and_si256 (
                         upperLsw,
                         _mm256_sllv_epi64(
                            vectorExtractionMask,
                            upperInDigitIdx_epi64
                         ) // end shift left
                      ),
                      upperInDigitIdx_epi64
                   ) // end shift right
                 );
#endif // P%DIGIT_SIZE_b < 8

#endif // AVX2

__asm__ __volatile__ (  "nop\n\t"
                        "nop\n\t"
                        "nop\n\t"
                     );

   *__lowerResult = lowerResult;
   *__upperResult = upperResult;
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

            // for(int vecSyndBitsElemIdx = 0;
            //    vecSyndBitsElemIdx < DIGIT_SIZE_B;
            //    vecSyndBitsElemIdx++){
            //       POSITION_T tmp = HtrPosOnes[i][vecSyndBitsWordIdx*DIGIT_SIZE_B+vecSyndBitsElemIdx];
            //       /*note: the position will be the one of the lowest exponent in the bitvector */
            //       tmp += vectorIdx * DIGIT_SIZE_B;
            //       tmp = tmp >= P ? tmp - P : tmp;
            //       vecSyndBits[vecSyndBitsWordIdx] = ROTBYTE(vecSyndBits[vecSyndBitsWordIdx]);
            //       vecSyndBits[vecSyndBitsWordIdx] += gf2x_get_8_coeff_vector(currSyndrome,tmp);
            //    }

#ifdef HIGH_COMPATIBILITY_X86_64 // MMX to AVX2
            // 8x uint32_t of HtrPosOnes[i] are loaded in a 256-bit vector unit as single precision float
            __m256i tmpReg = _mm256_castps_si256(_mm256_loadu_ps((float*) &HtrPosOnes[i][vecSyndBitsWordIdx*DIGIT_SIZE_B]));
            __m256  cmpMask; // used in the conditional assignment
            __m256  tmpResultOfSubtraction; // used in the conditional assignment, it's the result of tmp - P

#ifdef HIGH_PERFORMANCE_X86_64 // AVX2 only
            // semantically equivalent to tmp += vectorIdx * DIGIT_SIZE_B
            __m256i addend = _mm256_set1_epi32 (vectorIdx * DIGIT_SIZE_B);
            tmpReg = _mm256_add_epi32 (tmpReg, addend);

            /* I need to compare tmpReg and broadcastedP to check if
             * each element in tmpReg is greather-than-or-equal to P.
             * To do that, a compare mask will be composed by a greather-than comparision
             * between P and tmpReg (P > tmpRegElement) so that if it's true
             * the element mustn't change. */
             __m256i broadcastedP = _mm256_set1_epi32(P);
             tmpResultOfSubtraction = _mm256_castsi256_ps(_mm256_sub_epi32 (tmpReg, broadcastedP));

            // __m256i equalMask = _mm256_cmpeq_epi32 (__m256i a, __m256i b);
            // __m256i greatherThanMask = _mm256_cmpgt_epi32 (__m256i a, __m256i b);
            //
            // __m256i cmpMask = __m256i _mm256_or_si256 (equalMask, greatherThanMask);

            cmpMask = _mm256_castsi256_ps(_mm256_cmpgt_epi32 (broadcastedP, tmpReg)); // m256i cast to m256

#else // MMX to AVX only

            // tmpReg splitted in two 128i vectors
            __m128i lowerTmp = _mm256_extractf128i_lower(tmpReg);
            __m128i upperTmp = _mm256_extractf128i_upper(tmpReg);

            // semantically equivalent to tmp += vectorIdx * DIGIT_SIZE_B
            __m128i addend = _mm_set1_epi32(vectorIdx * DIGIT_SIZE_B); // vectorIdx * DIGIT_SIZE_B saved in a __m128i
            lowerTmp = _mm_add_epi32(lowerTmp, addend);
            upperTmp = _mm_add_epi32(upperTmp, addend);

            tmpReg = _mm256_setr_m128 (lowerTmp, upperTmp);

            __m128i broadcastedP = _mm_set1_epi32 (P);
            tmpResultOfSubtraction = _mm256_setr_m128 (_mm_sub_epi32 (lowerTmp, broadcastedP), _mm_sub_epi32 (upperTmp, broadcastedP));

            // // I want to compare each element of tmpReg with P: here I'm using a compare function that will
            // // return a mask (0x1d = 29 means Greater-than-or-equal (ordered, non-signaling))
            //
            // __m256 cmpMask = _mm256_cmp_ps(tmpReg, broadcastedP, 0x1d);

            __m128i upperCmpMask = _mm_cmpgt_epi32 (broadcastedP, upperTmp);
            __m128i lowerCmpMask = _mm_cmpgt_epi32 (broadcastedP, lowerTmp);
            cmpMask = _mm256_castsi256_ps(_mm256_setr_m128i(lowerCmpMask, upperCmpMask));

#endif // end of (AVX2 only) and (MMX to AVX only)

            // semantically equivalent to: tmp = tmp >= P ? tmp - P : tmp;
            // changed to tmp = P > tmp ? tmp : tmpResultOfSubtraction
            tmpReg = _mm256_blendv_ps (tmpResultOfSubtraction, _mm256_castsi256_ps(tmpReg), cmpMask);

            __m256i lowerResult, upperResult;

            get_64_coeff_vector(currSyndrome, tmpReg, &lowerResult, &upperResult);
            // Raccolgo gli ultimi 8 bit del risultato ma è in ordine inverso rispetto a prima (?)

#endif // end of MMX to AVX2

         } // end for vecSyndBitsWordIdx

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
            } // end for vectorIdx
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
