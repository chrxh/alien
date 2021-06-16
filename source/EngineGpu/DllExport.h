#pragma once

#include <QtCore/qglobal.h>

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef ENGINEGPU_LIB
# define ENGINEGPU_EXPORT Q_DECL_EXPORT
#else
# define ENGINEGPU_EXPORT Q_DECL_IMPORT
#endif
#else
# define ENGINEGPU_EXPORT
#endif
