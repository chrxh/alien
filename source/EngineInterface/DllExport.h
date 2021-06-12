#pragma once

#include <QtCore/qglobal.h>

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef ENGINEINTERFACE_LIB
# define ENGINEINTERFACE_EXPORT Q_DECL_EXPORT
#else
# define ENGINEINTERFACE_EXPORT Q_DECL_IMPORT
#endif
#else
# define ENGINEINTERFACE_EXPORT
#endif
