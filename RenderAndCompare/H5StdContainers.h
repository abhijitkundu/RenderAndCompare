/**
 * @file H5StdContainers.h
 * @brief H5StdContainers
 *
 * @author Abhijit Kundu
 */

#ifndef H5STDCONTAINERS_H_
#define H5STDCONTAINERS_H_

#include <string>
#include <vector>

#include <H5Cpp.h>

namespace H5StdContainers {

void save(const H5::CommonFG& h5fg, const std::string& dsname, const std::vector<std::string>& strings);
void save(const H5::CommonFG& h5fg, const std::string& dsname, const std::string& string);


void load(const H5::CommonFG& h5fg, const std::string& dsname, std::vector<std::string>& strings);
void load(const H5::CommonFG& h5fg, const std::string& dsname, std::string& string);

std::vector<std::string> read_vector_of_strings(const H5::CommonFG& h5fg, const std::string& dsname);

}  // end namespace H5StdContainers

#endif // end H5STDCONTAINERS_H_
