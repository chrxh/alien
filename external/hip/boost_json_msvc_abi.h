#pragma once

// Force-included into the Boost::json consumers of the Windows HIP build (clang-cl).
//
// The vcpkg Boost.JSON library is compiled with MSVC, which declares
// default_resource::instance_ as "holder". clang declares it as
// "[[clang::no_destroy]] default_resource" instead, and since the MSVC name
// mangling encodes the type, the import of instance_ fails to link. Undefining
// the macro after the config header gives clang-cl the declaration the library
// was built with.

#include <boost/json/detail/config.hpp>

#undef BOOST_JSON_NO_DESTROY
