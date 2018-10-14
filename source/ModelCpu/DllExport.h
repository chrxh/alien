#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef MODELCPU_LIB
# define MODELCPU_EXPORT Q_DECL_EXPORT
#else
# define MODELCPU_EXPORT Q_DECL_IMPORT
#endif
#else
# define MODELCPU_EXPORT
#endif
