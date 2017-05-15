/**
 * @file EigenTypedefs.h
 * @brief EigenTypedefs
 *
 * @author Abhijit Kundu
 */

#ifndef EIGEN_TYPEDEFS_H_
#define EIGEN_TYPEDEFS_H_

#include <Eigen/StdVector>
#include <map>

namespace Eigen {

/**@brief Eigen Aligned StdVector
 *
 * Convenience alias template typedef for std::vector with Eigen::aligned_allocator
 *
 * Usage: Eigen::AlignedStdVector<Eigen::Isometery3d> my_data;
 *
 */
template<class T>
using AlignedStdVector = std::vector<T, aligned_allocator<T> >;

/**@brief Eigen Aligned Map container
 *
 * Convenience alias template typedef for std::map with Eigen::aligned_allocator
 *
 */
template<class Key, class T, class Compare = std::less<Key>>
using AlignedStdMap = std::map<Key, T, Compare, aligned_allocator<std::pair<const Key,T>>>;

}  // namespace Eigen

#endif // end EIGEN_TYPEDEFS_H_
