#ifndef WEB_GLOBAL_H
#define WEB_GLOBAL_H

#include <QtCore/qglobal.h>

#ifdef WEB_LIB
# define WEB_EXPORT Q_DECL_EXPORT
#else
# define WEB_EXPORT Q_DECL_IMPORT
#endif

#endif // WEB_GLOBAL_H
