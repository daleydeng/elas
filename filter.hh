#ifndef __FILTER_H__
#define __FILTER_H__

#include <stdint.h>

// fast filters: implements 3x3 and 5x5 sobel filters and
//               5x5 blob and corner filters based on SSE2/3 instructions
namespace filter {

void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );

void sobel5x5( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h );

// -1 -1  0  1  1
// -1 -1  0  1  1
//  0  0  0  0  0
//  1  1  0 -1 -1
//  1  1  0 -1 -1
void checkerboard5x5( const uint8_t* in, int16_t* out, int w, int h );

// -1 -1 -1 -1 -1
// -1  1  1  1 -1
// -1  1  8  1 -1
// -1  1  1  1 -1
// -1 -1 -1 -1 -1
void blob5x5( const uint8_t* in, int16_t* out, int w, int h );
};

#endif
