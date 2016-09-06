#ifndef __FILTER_H__
#define __FILTER_H__

#include <stdint.h>

namespace elas {

void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );

int sad_u8(const void *a, const void *b);
void updatePosteriorMinimum(void* I2_block_addr, int32_t d, int32_t w, const void *xmm1, void *xmm2, int32_t &val, int32_t &min_val, int32_t &min_d);
void updatePosteriorMinimum (void* I2_block_addr, int32_t d, const void *xmm1, void *xmm2, int32_t &val, int32_t &min_val, int32_t &min_d);

};

#endif
