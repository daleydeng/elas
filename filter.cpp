#include <stdio.h>
#include <string.h>
#include <cassert>
#include <stdint.h>
#include <simdpp/simd.h>
#include <iostream>
#include <glog/logging.h>
#include "filter.hh"

// fast filters: implements 3x3 and 5x5 sobel filters and
//               5x5 blob and corner filters based on SSE2/3 instructions
namespace filter {

using namespace simdpp;

static const int SIZE16 = SIMDPP_FAST_INT16_SIZE;
static const int SIZE8 = SIMDPP_FAST_INT8_SIZE;

void check_width(int w)
{
  CHECK(w % SIZE8 == 0) << "width must be multiple of "<<SIZE8;
}
// private namespace, public user functions at the bottom of this file
namespace detail {
void integral_image( const uint8_t* in, int32_t* out, int w, int h ) {
  int32_t* out_top = out;
  const uint8_t* line_end = in + w;
  const uint8_t* in_end   = in + w*h;
  int32_t line_sum = 0;
  for( ; in != line_end; in++, out++ ) {
    line_sum += *in;
    *out = line_sum;
  }
  for( ; in != in_end; ) {
    int32_t line_sum = 0;
    const uint8_t* line_end = in + w;
    for( ; in != line_end; in++, out++, out_top++ ) {
      line_sum += *in;
      *out = *out_top + line_sum;
    }
  }
}

void unpack_8bit_to_16bit( const __m128i a, __m128i& b0, __m128i& b1 ) {
  __m128i zero = _mm_setzero_si128();
  b0 = _mm_unpacklo_epi8( a, zero );
  b1 = _mm_unpackhi_epi8( a, zero );
}

void unpack_8bit_to_16bit(const void *p, void *b0, void *b1) {
  const auto *d0 = (uint8x16*)p;
  uint8x16 zero = make_int(0);
  *((uint8x16*)b0) = simdpp::zip16_lo(*d0, zero);
  *((uint8x16*)b1) = simdpp::zip16_hi(*d0, zero);
}

void pack_16bit_to_8bit_saturate( const __m128i a0, const __m128i a1, __m128i& b ) {
  b = _mm_packus_epi16( a0, a1 );
}

// convolve image with a (1,4,6,4,1) row vector. Result is accumulated into output.
// output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_14641_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
  check_width(w);
  const int16_t *i0 = in, *i1 = in+1, *i2 = in+2, *i3 = in+3, *i4 = in+4;
  uint8_t* result   = out + 2;
  const int16_t* end_input = in + w*h;
  int16x8 offs = make_int(128);
  for( ; i4 < end_input; i0 += 8, i1 += 8, i2 += 8, i3 += 8, i4 += 8, result += 16 ) {
    __m128i ret_lo;
    __m128i ret_hi;
    for( int i=0; i<2; i++ ) {
      __m128i* ret;
      if( i==0 )
        ret = &ret_lo;
      else
        ret = &ret_hi;

      __m128i i0_register = _mm_loadu_si128((__m128i*)i0);
      __m128i i1_register = _mm_loadu_si128( (__m128i*)( i1 ) );
      __m128i i2_register = _mm_loadu_si128( (__m128i*)( i2 ) );
      __m128i i3_register = _mm_loadu_si128( (__m128i*)( i3 ) );
      __m128i i4_register = _mm_loadu_si128( (__m128i*)( i4 ) );
      *ret = _mm_setzero_si128();
      *ret = _mm_add_epi16( i0_register, *ret );
      i1_register      = _mm_add_epi16( i1_register, i1_register  );
      i1_register      = _mm_add_epi16( i1_register, i1_register  );
      *ret = _mm_add_epi16( i1_register, *ret );
      i2_register      = _mm_add_epi16( i2_register, i2_register  );
      *ret = _mm_add_epi16( i2_register, *ret );
      i2_register      = _mm_add_epi16( i2_register, i2_register  );
      *ret = _mm_add_epi16( i2_register, *ret );
      i3_register      = _mm_add_epi16( i3_register, i3_register  );
      i3_register      = _mm_add_epi16( i3_register, i3_register  );
      *ret = _mm_add_epi16( i3_register, *ret );
      *ret = _mm_add_epi16( i4_register, *ret );
      *ret = _mm_srai_epi16( *ret, 7 );
      *ret = _mm_add_epi16( *ret, offs );
      if( i==0 ) {
        i0 += 8;
        i1 += 8;
        i2 += 8;
        i3 += 8;
        i4 += 8;
      }
    }
    pack_16bit_to_8bit_saturate( ret_lo, ret_hi, ret_lo );
    _mm_storeu_si128( ((__m128i*)( result )), ret_lo );
  }
}

// convolve image with a (1,2,0,-2,-1) row vector. Result is accumulated into output.
// This one works on 16bit input and 8bit output.
// output is scaled by 1/128, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_12021_row_5x5_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
  check_width(w);
  const __m128i*  i0 = (const __m128i*)(in);
  const int16_t*    i1 = in+1;
  const int16_t*    i3 = in+3;
  const int16_t*    i4 = in+4;
  uint8_t* result    = out + 2;
  const int16_t* const end_input = in + w*h;
  __m128i offs = _mm_set1_epi16( 128 );
  for( ; i4 < end_input; i0 += 1, i1 += 8, i3 += 8, i4 += 8, result += 16 ) {
    __m128i ret_lo;
    __m128i ret_hi;
    for( int i=0; i<2; i++ ) {
      __m128i* ret;
      if( i==0 ) ret = &ret_lo;
      else       ret = &ret_hi;
      __m128i i0_register = *i0;
      __m128i i1_register = _mm_loadu_si128( (__m128i*)( i1 ) );
      __m128i i3_register = _mm_loadu_si128( (__m128i*)( i3 ) );
      __m128i i4_register = _mm_loadu_si128( (__m128i*)( i4 ) );
      *ret = _mm_setzero_si128();
      *ret = _mm_add_epi16( i0_register,   *ret );
      i1_register      = _mm_add_epi16( i1_register, i1_register  );
      *ret = _mm_add_epi16( i1_register,   *ret );
      i3_register      = _mm_add_epi16( i3_register, i3_register  );
      *ret = _mm_sub_epi16( *ret, i3_register );
      *ret = _mm_sub_epi16( *ret, i4_register );
      *ret = _mm_srai_epi16( *ret, 7 );
      *ret = _mm_add_epi16( *ret, offs );
      if( i==0 ) {
        i0 += 1;
        i1 += 8;
        i3 += 8;
        i4 += 8;
      }
    }
    pack_16bit_to_8bit_saturate( ret_lo, ret_hi, ret_lo );
    _mm_storeu_si128( ((__m128i*)( result )), ret_lo );
  }
}

// convolve image with a (1,2,1) row vector. Result is accumulated into output.
// This one works on 16bit input and 8bit output.
// output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_121_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
  check_width(w);
  const __m128i* i0 = (const __m128i*)(in);
  const int16_t* i1 = in+1;
  const int16_t* i2 = in+2;
  uint8_t* result   = out + 1;
  const size_t blocked_loops = (w*h-2)/16;
  __m128i offs = _mm_set1_epi16( 128 );
  for( size_t i=0; i != blocked_loops; i++ ) {
    __m128i ret_lo;
    __m128i ret_hi;
    __m128i i1_register;
    __m128i i2_register;

    i1_register        = _mm_loadu_si128( (__m128i*)( i1 ) );
    i2_register        = _mm_loadu_si128( (__m128i*)( i2 ) );
    ret_lo = *i0;
    i1_register        = _mm_add_epi16( i1_register, i1_register );
    ret_lo = _mm_add_epi16( i1_register, ret_lo );
    ret_lo = _mm_add_epi16( i2_register, ret_lo );
    ret_lo = _mm_srai_epi16( ret_lo, 2 );
    ret_lo = _mm_add_epi16( ret_lo, offs );

    i0++;
    i1+=8;
    i2+=8;

    i1_register        = _mm_loadu_si128( (__m128i*)( i1 ) );
    i2_register        = _mm_loadu_si128( (__m128i*)( i2 ) );
    ret_hi = *i0;
    i1_register        = _mm_add_epi16( i1_register, i1_register );
    ret_hi = _mm_add_epi16( i1_register, ret_hi );
    ret_hi = _mm_add_epi16( i2_register, ret_hi );
    ret_hi = _mm_srai_epi16( ret_hi, 2 );
    ret_hi = _mm_add_epi16( ret_hi, offs );

    i0++;
    i1+=8;
    i2+=8;

    pack_16bit_to_8bit_saturate( ret_lo, ret_hi, ret_lo );
    _mm_storeu_si128( ((__m128i*)( result )), ret_lo );

    result += 16;
  }
}

// convolve image with a (1,0,-1) row vector. Result is accumulated into output.
// This one works on 16bit input and 8bit output.
// output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_101_row_3x3_16bit( const int16_t* in, uint8_t* out, int w, int h ) {
  check_width(w);
  const __m128i*  i0 = (const __m128i*)(in);
  const int16_t*    i2 = in+2;
  uint8_t* result    = out + 1;
  const int16_t* const end_input = in + w*h;
  const size_t blocked_loops = (w*h-2)/16;
  __m128i offs = _mm_set1_epi16( 128 );
  for( size_t i=0; i != blocked_loops; i++ ) {
    __m128i ret_lo;
    __m128i ret_hi;
    __m128i i2_register;

    i2_register = _mm_loadu_si128( (__m128i*)( i2 ) );
    ret_lo  = *i0;
    ret_lo  = _mm_sub_epi16( ret_lo, i2_register );
    ret_lo  = _mm_srai_epi16( ret_lo, 2 );
    ret_lo  = _mm_add_epi16( ret_lo, offs );

    i0 += 1;
    i2 += 8;

    i2_register = _mm_loadu_si128( (__m128i*)( i2 ) );
    ret_hi  = *i0;
    ret_hi  = _mm_sub_epi16( ret_hi, i2_register );
    ret_hi  = _mm_srai_epi16( ret_hi, 2 );
    ret_hi  = _mm_add_epi16( ret_hi, offs );

    i0 += 1;
    i2 += 8;

    pack_16bit_to_8bit_saturate( ret_lo, ret_hi, ret_lo );
    _mm_storeu_si128( ((__m128i*)( result )), ret_lo );

    result += 16;
  }

  for( ; i2 < end_input; i2++, result++) {
    *result = ((*(i2-2) - *i2)>>2)+128;
  }
}

void convolve_cols_5x5( const unsigned char* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
  check_width(w);
  using namespace std;
  memset( out_h, 0, w*h*sizeof(int16_t) );
  memset( out_v, 0, w*h*sizeof(int16_t) );
  const int w_chunk  = w/16;
  __m128i*  i0       = (__m128i*)( in );
  __m128i*  i1       = (__m128i*)( in ) + w_chunk*1;
  __m128i*  i2       = (__m128i*)( in ) + w_chunk*2;
  __m128i*  i3       = (__m128i*)( in ) + w_chunk*3;
  __m128i*  i4       = (__m128i*)( in ) + w_chunk*4;
  __m128i* result_h  = (__m128i*)( out_h ) + 4*w_chunk;
  __m128i* result_v  = (__m128i*)( out_v ) + 4*w_chunk;
  __m128i* end_input = (__m128i*)( in ) + w_chunk*h;
  __m128i sixes      = _mm_set1_epi16( 6 );
  __m128i fours      = _mm_set1_epi16( 4 );
  for( ; i4 != end_input; i0++, i1++, i2++, i3++, i4++, result_v+=2, result_h+=2 ) {
    __m128i ilo, ihi;
    unpack_8bit_to_16bit( *i0, ihi, ilo );
    *result_h     = _mm_add_epi16( ihi, *result_h );
    *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
    *result_v     = _mm_add_epi16( *result_v, ihi );
    *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
    unpack_8bit_to_16bit( *i1, ihi, ilo );
    *result_h     = _mm_add_epi16( ihi, *result_h );
    *result_h     = _mm_add_epi16( ihi, *result_h );
    *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
    *(result_h+1) = _mm_add_epi16( ilo, *(result_h+1) );
    ihi = _mm_mullo_epi16( ihi, fours );
    ilo = _mm_mullo_epi16( ilo, fours );
    *result_v     = _mm_add_epi16( *result_v, ihi );
    *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
    unpack_8bit_to_16bit( *i2, ihi, ilo );
    ihi = _mm_mullo_epi16( ihi, sixes );
    ilo = _mm_mullo_epi16( ilo, sixes );
    *result_v     = _mm_add_epi16( *result_v, ihi );
    *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
    unpack_8bit_to_16bit( *i3, ihi, ilo );
    *result_h     = _mm_sub_epi16( *result_h, ihi );
    *result_h     = _mm_sub_epi16( *result_h, ihi );
    *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
    *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
    ihi = _mm_mullo_epi16( ihi, fours );
    ilo = _mm_mullo_epi16( ilo, fours );
    *result_v     = _mm_add_epi16( *result_v, ihi );
    *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
    unpack_8bit_to_16bit( *i4, ihi, ilo );
    *result_h     = _mm_sub_epi16( *result_h, ihi );
    *(result_h+1) = _mm_sub_epi16( *(result_h+1), ilo );
    *result_v     = _mm_add_epi16( *result_v, ihi );
    *(result_v+1) = _mm_add_epi16( *(result_v+1), ilo );
  }
}

void convolve_col_p1p1p0m1m1_5x5( const unsigned char* in, int16_t* out, int w, int h ) {
  check_width(w);
  memset( out, 0, w*h*sizeof(int16_t) );
  using namespace std;
  const int w_chunk  = w/16;
  __m128i*  i0       = (__m128i*)( in );
  __m128i*  i1       = (__m128i*)( in ) + w_chunk*1;
  __m128i*  i3       = (__m128i*)( in ) + w_chunk*3;
  __m128i*  i4       = (__m128i*)( in ) + w_chunk*4;
  __m128i* result    = (__m128i*)( out ) + 4*w_chunk;
  __m128i* end_input = (__m128i*)( in ) + w_chunk*h;
  for( ; i4 != end_input; i0++, i1++, i3++, i4++, result+=2 ) {
    __m128i ilo0, ihi0;
    unpack_8bit_to_16bit( *i0, ihi0, ilo0 );
    __m128i ilo1, ihi1;
    unpack_8bit_to_16bit( *i1, ihi1, ilo1 );
    *result     = _mm_add_epi16( ihi0, ihi1 );
    *(result+1) = _mm_add_epi16( ilo0, ilo1 );
    __m128i ilo, ihi;
    unpack_8bit_to_16bit( *i3, ihi, ilo );
    *result     = _mm_sub_epi16( *result, ihi );
    *(result+1) = _mm_sub_epi16( *(result+1), ilo );
    unpack_8bit_to_16bit( *i4, ihi, ilo );
    *result     = _mm_sub_epi16( *result, ihi );
    *(result+1) = _mm_sub_epi16( *(result+1), ilo );
  }
}

void convolve_row_p1p1p0m1m1_5x5(const int16_t* in, int16_t* out, int w, int h) {
  check_width(w);
  const auto* i0 = in;
  const auto* i1 = in+1;
  const auto* i3 = in+3;
  const auto* i4 = in+4;
  const auto* end_input = in + w*h;
  auto* result = out + 2;

  for(; i4+SIZE16 < end_input; i0 += SIZE16, i1 += SIZE16, i3 += SIZE16, i4 += SIZE16, result += SIZE16 ) {
    int16x8 ret, v_i0 = load(i0), v_i1 = load(i1), v_i3 = load(i3), v_i4 = load(i4);
    ret = simdpp::add(ret, v_i0);
    ret = simdpp::add(ret, v_i1);
    ret = simdpp::sub(ret, v_i3);
    ret = simdpp::sub(ret, v_i4);
    store(result, ret);
  }
}

void convolve_cols_3x3( const uint8_t* in, int16_t* out_v, int16_t* out_h, int w, int h ) {
  check_width(w);
  const uint8_t*  i0       = in;
  const uint8_t*  i1       = in + w;
  const uint8_t*  i2       = in + 2*w;
  const uint8_t* end_input =  in + w*h;
  auto *result_h_ = (int16x8*)(out_h + w);
  auto *result_v_ = (int16x8*)(out_v + w);
  auto zero_ = make_int(0);

  for( ; i2 != end_input; i0 += 16, i1 += 16, i2 += 16, result_h_+=2, result_v_+=2 ) {
    result_h_[0] = zero_;
    result_h_[1] = zero_;
    result_v_[0] = zero_;
    result_v_[1] = zero_;

    int16x8 ihi_, ilo_;
    unpack_8bit_to_16bit(i0, &ihi_, &ilo_);
    result_h_[0] = simdpp::add(result_h_[0], ihi_);
    result_h_[1] = simdpp::add(result_h_[1], ilo_);
    result_v_[0] = simdpp::add(result_v_[0], ihi_);
    result_v_[1] = simdpp::add(result_v_[1], ilo_);

    unpack_8bit_to_16bit(i1, &ihi_, &ilo_);
    result_v_[0] = simdpp::add(result_v_[0], ihi_);
    result_v_[1] = simdpp::add(result_v_[1], ilo_);

    result_v_[0] = simdpp::add(result_v_[0], ihi_);
    result_v_[1] = simdpp::add(result_v_[1], ilo_);

    unpack_8bit_to_16bit(i2, &ihi_, &ilo_);
    result_h_[0] = simdpp::sub(result_h_[0], ihi_);
    result_h_[1] = simdpp::sub(result_h_[1], ilo_);

    result_v_[0] = simdpp::add(result_v_[0], ihi_);
    result_v_[1] = simdpp::add(result_v_[1], ilo_);
  }
}

}

void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h ) {
  int16_t* temp_h = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
  int16_t* temp_v = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
  detail::convolve_cols_3x3( in, temp_v, temp_h, w, h );
  detail::convolve_101_row_3x3_16bit( temp_v, out_v, w, h );
  detail::convolve_121_row_3x3_16bit( temp_h, out_h, w, h );
  _mm_free( temp_h );
  _mm_free( temp_v );
}

void sobel5x5( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h ) {
  int16_t* temp_h = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
  int16_t* temp_v = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
  detail::convolve_cols_5x5( in, temp_v, temp_h, w, h );
  detail::convolve_12021_row_5x5_16bit( temp_v, out_v, w, h );
  detail::convolve_14641_row_5x5_16bit( temp_h, out_h, w, h );
  _mm_free( temp_h );
  _mm_free( temp_v );
}

// -1 -1  0  1  1
// -1 -1  0  1  1
//  0  0  0  0  0
//  1  1  0 -1 -1
//  1  1  0 -1 -1
void checkerboard5x5( const uint8_t* in, int16_t* out, int w, int h ) {
  int16_t* temp = (int16_t*)( _mm_malloc( w*h*sizeof( int16_t ), 16 ) );
  detail::convolve_col_p1p1p0m1m1_5x5( in, temp, w, h );
  detail::convolve_row_p1p1p0m1m1_5x5( temp, out, w, h );
  _mm_free( temp );
}

// -1 -1 -1 -1 -1
// -1  1  1  1 -1
// -1  1  8  1 -1
// -1  1  1  1 -1
// -1 -1 -1 -1 -1
void blob5x5( const uint8_t* in, int16_t* out, int w, int h ) {
  int32_t* integral = (int32_t*)( _mm_malloc( w*h*sizeof( int32_t ), 16 ) );
  detail::integral_image( in, integral, w, h );
  int16_t* out_ptr   = out + 3 + 3*w;
  int16_t* out_end   = out + w * h - 2 - 2*w;
  const int32_t* i00 = integral;
  const int32_t* i50 = integral + 5;
  const int32_t* i05 = integral + 5*w;
  const int32_t* i55 = integral + 5 + 5*w;
  const int32_t* i11 = integral + 1 + 1*w;
  const int32_t* i41 = integral + 4 + 1*w;
  const int32_t* i14 = integral + 1 + 4*w;
  const int32_t* i44 = integral + 4 + 4*w;
  const uint8_t* im22 = in + 3 + 3*w;
  for( ; out_ptr != out_end; out_ptr++, i00++, i50++, i05++, i55++, i11++, i41++, i14++, i44++, im22++ ) {
    int32_t result = 0;
    result = -( *i55 - *i50 - *i05 + *i00 );
    result += 2*( *i44 - *i41 - *i14 + *i11 );
    result += 7* *im22;
    *out_ptr = result;
  }
  _mm_free( integral );
}
};
