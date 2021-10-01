#pragma once

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef ENGINEIMPL_LIB
# define ENGINEIMPL_EXPORT __declspec(dllexport)
#else
# define ENGINEIMPL_EXPORT __declspec(dllimport)
#endif
#else
# define ENGINEIMPL_EXPORT
#endif
