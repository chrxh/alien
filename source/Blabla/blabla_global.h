#ifndef BLABLA_GLOBAL_H
#define BLABLA_GLOBAL_H

#include <QtCore/qglobal.h>

#ifdef BLABLA_LIB
# define BLABLA_EXPORT Q_DECL_EXPORT
#else
# define BLABLA_EXPORT Q_DECL_IMPORT
#endif

#endif // BLABLA_GLOBAL_H
