#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef WEB_LIB
# define WEB_EXPORT Q_DECL_EXPORT
#else
# define WEB_EXPORT Q_DECL_IMPORT
#endif
#else
# define WEB_EXPORT
#endif
