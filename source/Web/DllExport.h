#pragma once

#include <QtCore/qglobal.h>

#if defined(_WIN32) && !defined(ALIEN_STATIC)
#ifdef WEB_LIB
# define WEB_EXPORT Q_DECL_EXPORT
#else
# define WEB_EXPORT Q_DECL_IMPORT
#endif
#else
# define WEB_EXPORT
#endif
