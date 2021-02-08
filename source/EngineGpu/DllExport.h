#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef ENGINEGPU_LIB
# define ENGINEGPU_EXPORT Q_DECL_EXPORT
#else
# define ENGINEGPU_EXPORT Q_DECL_IMPORT
#endif
#else
# define ENGINEGPU_EXPORT
#endif
