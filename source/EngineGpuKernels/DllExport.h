#pragma once

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef ENGINEGPUKERNELS_LIB
#define ENGINEGPUKERNELS_EXPORT __declspec(dllexport)
#else
#define ENGINEGPUKERNELS_EXPORT __declspec(dllimport)
#endif
#else
#define ENGINEGPUKERNELS_EXPORT
#endif
