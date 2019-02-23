typedef struct {
  __m128i multiplier;       // multiplier used in fast division
  __m128i shift1;           // shift count 1 used in fast division
  __m128i shift2;           // shift count 2 used in fast division
} Divisor_ui;

static inline Divisor_ui * getDivisor(uint32_t d) {
  uint32_t L, L2, sh1, sh2, m;
  switch (d) {
  case 0:
      m = sh1 = sh2 = 1 / d;                         // provoke error for d = 0
      break;
  case 1:
      m = 1; sh1 = sh2 = 0;                          // parameters for d = 1
      break;
  case 2:
      m = 1; sh1 = 1; sh2 = 0;                       // parameters for d = 2
      break;
  default:                                           // general case for d > 2
      L   = bit_scan_reverse(d-1)+1;                  // ceil(log2(d))
      L2  = L < 32 ? 1 << L : 0;                      // 2^L, overflow to 0 if L = 32
      m   = 1 + uint32_t((uint64_t(L2 - d) << 32) / d); // multiplier
      sh1 = 1;  sh2 = L - 1;                         // shift counts
  }

  Divisor_ui *divisor   = (Divisor_ui *) calloc(sizeof(Divisor_ui));
  
  (*divisor).multiplier = _mm_set1_epi32(m);
  (*divisor).shift1     = _mm_setr_epi32(sh1, 0, 0, 0);
  (*divisor).shift2     = _mm_setr_epi32(sh2, 0, 0, 0);

  return divisor;
}

static inline __m128i _mm_div_epi32(__m128i dividend, uint32_t d) {
  // divide every uint32 word by divisor
  if (d == 1) return dividend;

  Divisor_ui *divisor = getDivisor(d);

  __m128i t1  = _mm_mul_epu32(a,(*divisor).multiplier);  // 32x32->64 bit unsigned multiplication of a[0] and a[2]
  __m128i t2  = _mm_srli_epi64(t1,32);                   // high dword of result 0 and 2
  __m128i t3  = _mm_srli_epi64(a,32);                    // get a[1] and a[3] into position for multiplication
  __m128i t4  = _mm_mul_epu32(t3,(*divisor).multiplier); // 32x32->64 bit unsigned multiplication of a[1] and a[3]
  __m128i t5  = _mm_set_epi32(-1,0,-1,0);                // mask of dword 1 and 3
#if INSTRSET >= 5   // SSE4.1 supported
  __m128i t7  = _mm_blendv_epi8(t2,t4,t5);               // blend two results
#else
  __m128i t6  = _mm_and_si128(t4,t5);                    // high dword of result 1 and 3
  __m128i t7  = _mm_or_si128(t2,t6);                     // combine all four results into one vector
#endif
  __m128i t8  = _mm_sub_epi32(a,t7);                     // subtract
  __m128i t9  = _mm_srl_epi32(t8,(*divisor).shift1);     // shift right logical
  __m128i t10 = _mm_add_epi32(t7,t9);                    // add
  free(divisor);
  return        _mm_srl_epi32(t10,(*divisor).shift2);    // shift right logical
}
