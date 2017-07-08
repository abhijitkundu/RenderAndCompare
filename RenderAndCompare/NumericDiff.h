/**
 * @file NumericDiff.h
 * @brief NumericDiff
 *
 * @author Abhijit Kundu
 */

#ifndef RENDERANDCOMPARE_NUMERIC_DIFF_H_
#define RENDERANDCOMPARE_NUMERIC_DIFF_H_

namespace RaC {

template <class T>
void central_diff_gpu(const int n,
                      const T* const fplus,
                      const T* const fminus,
                      const T step_size,
                      T* diff,
                      const int diff_stride);

}  // namespace RaC

#endif // end RENDERANDCOMPARE_NUMERIC_DIFF_H_
