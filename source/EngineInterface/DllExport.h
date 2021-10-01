#pragma once

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef ENGINEINTERFACE_LIB
# define ENGINEINTERFACE_EXPORT __declspec(dllexport)
#else
# define ENGINEINTERFACE_EXPORT __declspec(dllimport)
#endif
#else
# define ENGINEINTERFACE_EXPORT
#endif
