#pragma once
#include "qc_ldpc_parameters.h"
#include "gf2x_limbs.h"

#if DV % DIGIT_SIZE_B == 0
#define REM_PADD_IMM8_L 0x0
#define REM_PADD_IMM8_H 0x0


#elif DV % DIGIT_SIZE_B == 1
#define REM_PADD_IMM8_L 0x1
#define REM_PADD_IMM8_H 0x0


#elif DV % DIGIT_SIZE_B == 2
#define REM_PADD_IMM8_L 0x3
#define REM_PADD_IMM8_H 0x0


#elif DV % DIGIT_SIZE_B == 3
#define REM_PADD_IMM8_L 0x7
#define REM_PADD_IMM8_H 0x0

#elif DV % DIGIT_SIZE_B == 4
#define REM_PADD_IMM8_L 0xF
#define REM_PADD_IMM8_H 0x0

#elif DV % DIGIT_SIZE_B == 5
#define REM_PADD_IMM8_L 0xF
#define REM_PADD_IMM8_H 0x1

#elif DV % DIGIT_SIZE_B == 6
#define REM_PADD_IMM8_L 0xF
#define REM_PADD_IMM8_H 0x3

#elif DV % DIGIT_SIZE_B == 7
#define REM_PADD_IMM8_L 0xF
#define REM_PADD_IMM8_H 0x7

#endif
