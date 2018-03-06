#ifndef MODEL_DLLEXPORT_H
#define MODEL_DLLEXPORT_H

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef MODEL_LIB
# define MODEL_EXPORT Q_DECL_EXPORT
#else
# define MODEL_EXPORT Q_DECL_IMPORT
#endif
#else
# define MODEL_EXPORT
#endif

#endif // MODEL_DLLEXPORT_H
