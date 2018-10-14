#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef MODELBASIC_LIB
# define MODELBASIC_EXPORT Q_DECL_EXPORT
#else
# define MODELBASIC_EXPORT Q_DECL_IMPORT
#endif
#else
# define MODELBASIC_EXPORT
#endif
