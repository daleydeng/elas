#include <stdio.h>
#include <string.h>
#include <cassert>
#include <stdint.h>
#include <simdpp/simd.h>
#include <iostream>
#include <glog/logging.h>
#include "common.hh"
#include <prettyprint.hh>

namespace elas {

using namespace simdpp;

namespace {

std::ostream &operator <<(std::ostream& os, const __m128i &v)
{
  auto *vv = (uint8_t*)(&v);
  os << (int)vv[0];
  for (int i = 1; i < 16; i++)
    os << " " << (int)vv[i];
  return os;
}

static const int SIZE16 = SIMDPP_FAST_INT16_SIZE;
static const int SIZE8 = SIMDPP_FAST_INT8_SIZE;

template<typename T>
static T simd_mul2(const T &v)
{
  return simdpp::add(v, v);
}

template<typename T>
static T simd_mul4(const T &v)
{
  T v2 = simd_mul2(v);
  return simdpp::add(v2, v2);
}

template<typename T>
static T simd_mul6(const T &v)
{
  T v4 = simd_mul4(v);
  T v2 = simdpp::add(v, v);
  return simdpp::add(v4, v2);
}

void check_width(int w)
{
  CHECK(w % SIZE8 == 0) << "width must be multiple of "<<SIZE8;
}

static int16x8 zero16 = make_int(0);
static int16x8 max16_1byte = make_int(255);

uint8x16 pack_16bit_to_8bit_saturate(const void *a0, const void *a1)
{
  int16x8 *a0_16 = (int16x8*)a0, *a1_16 = (int16x8*)a1;
  *a0_16 = simdpp::max(*a0_16, zero16);
  *a0_16 = simdpp::min(*a0_16, max16_1byte);
  *a1_16 = simdpp::max(*a1_16, zero16);
  *a1_16 = simdpp::min(*a1_16, max16_1byte);
  return simdpp::unzip16_lo(*(uint8x16*)a0_16, *(uint8x16*)a1_16);
}

void unpack_8bit_to_16bit(const void *p, void *b0, void *b1)
{
  const auto *d0 = (uint8x16*)p;
  uint8x16 zero = make_int(0);
  *((uint8x16*)b0) = simdpp::zip16_lo(*d0, zero);
  *((uint8x16*)b1) = simdpp::zip16_hi(*d0, zero);
}

// convolve image with a (1,2,1) row vector. Result is accumulated into output.
// This one works on 16bit input and 8bit output.
// output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_121_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h )
{
  const int16_t* i0 = in;
  const int16_t* i1 = in+1;
  const int16_t* i2 = in+2;
  uint8_t* result   = out + 1;
  int16x8 offs = make_int( 128 );

  for(int i = 0; i < (w*h-2)/SIZE8; i++, result += SIZE8, i0 += SIZE16, i1 += SIZE16, i2 += SIZE16) {
    int16x8 rlo = load_u(i0), rhi, v_i1 = load_u(i1), v_i2 = load_u(i2);
    v_i1 = simd_mul2(v_i1);
    rlo = simdpp::add(rlo, v_i1);
    rlo = simdpp::add(rlo, v_i2);
    rlo = simdpp::shift_r<2>(rlo);
    rlo = simdpp::add(rlo, offs);

    i0 += SIZE16;
    i1 += SIZE16;
    i2 += SIZE16;

    rhi = load_u(i0);
    v_i1 = load_u(i1);
    v_i1 = simd_mul2(v_i1);
    v_i2 = load_u(i2);
    rhi = simdpp::add(rhi, v_i1);
    rhi = simdpp::add(rhi, v_i2);
    rhi = simdpp::shift_r<2>(rhi);
    rhi = simdpp::add(rhi, offs);

    store_u(result, pack_16bit_to_8bit_saturate(&rlo, &rhi));
  }
}

// convolve image with a (1,0,-1) row vector. Result is accumulated into output.
// This one works on 16bit input and 8bit output.
// output is scaled by 1/4, then clamped to [-128,128], and finally shifted to [0,255].
void convolve_101_row_3x3_16bit(const int16_t* in, uint8_t* out, int w, int h )
{
  const int16_t*  i0 = in;
  const int16_t*    i2 = in+2;
  uint8_t* result    = out + 1;
  const int16_t* end_input = in + w*h;
  int16x8 offs = make_int( 128 );

  for(int i=0; i < (w*h-2)/SIZE8; i++, i0 += SIZE16, i2 += SIZE16, result += SIZE8) {
    int16x8 rlo = load_u(i0), rhi, v_i2 = load_u(i2);
    rlo = simdpp::sub(rlo, v_i2);
    rlo = simdpp::shift_r<2>(rlo);
    rlo = simdpp::add(rlo, offs);

    i0 += SIZE16;
    i2 += SIZE16;

    rhi = load_u(i0);
    v_i2 = load_u(i2);
    rhi = simdpp::sub(rhi, v_i2);
    rhi = simdpp::shift_r<2>(rhi);
    rhi = simdpp::add(rhi, offs);

    store_u(result, pack_16bit_to_8bit_saturate(&rlo, &rhi));
  }

  for( ; i2 < end_input; i2++, result++)
    *result = ((*(i2-2) - *i2)>>2)+128;
}

void convolve_cols_3x3( const uint8_t* in, int16_t* out_v, int16_t* out_h, int w, int h )
{
  const uint8_t*  i0       = in;
  const uint8_t*  i1       = in + w;
  const uint8_t*  i2       = in + 2*w;
  const uint8_t* end_input =  in + w*h;
  auto *result_h_ = (int16x8*)(out_h + w);
  auto *result_v_ = (int16x8*)(out_v + w);
  auto zero_ = make_int(0);

  for(; i2 != end_input; i0 += SIZE8, i1 += SIZE8, i2 += SIZE8, result_h_ += 2, result_v_+=2) {
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

} //namespace

void sobel3x3( const uint8_t* in, uint8_t* out_v, uint8_t* out_h, int w, int h ) {
  check_width(w);
  std::vector<int16_t, aligned_allocator<int16_t, SIZE8> > temp_h(w*h), temp_v(w*h);
  int16_t *ph = &temp_h[0], *pv = &temp_v[0];
  convolve_cols_3x3(in, pv, ph, w, h);
  convolve_101_row_3x3_16bit(pv, out_v, w, h);
  convolve_121_row_3x3_16bit(ph, out_h, w, h);
}

void updatePosteriorMinimum(void* I2_block_addr, int32_t d, int32_t w, const void *xmm1_, void *xmm2_, int32_t &val, int32_t &min_val, int32_t &min_d)
{
  int8x16 v1 = load(xmm1_), v2 = load(I2_block_addr);
  val = simdpp::reduce_add(simdpp::abs(simdpp::sub(v1, v2))) + w;
  if (val < min_val) {
    min_val = val;
    min_d   = d;
  }
}

void updatePosteriorMinimum (void* I2_block_addr, int32_t d, const void *xmm1_, void *xmm2_, int32_t &val, int32_t &min_val, int32_t &min_d)
{
  updatePosteriorMinimum(I2_block_addr, d, 0, xmm1_, xmm2_, val, min_val, min_d);
}

} // namespace elas
