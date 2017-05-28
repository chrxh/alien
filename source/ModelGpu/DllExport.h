#ifndef MODELGPU_DLLEXPORT_H
#define MODELGPU_DLLEXPORT_H

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef MODELGPU_LIB
# define MODELGPU_EXPORT Q_DECL_EXPORT
#else
# define MODELGPU_EXPORT Q_DECL_IMPORT
#endif
#else
# define MODELGPU_EXPORT
#endif

#endif // MODELGPU_DLLEXPORT_H
