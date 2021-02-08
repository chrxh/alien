#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef ENGINEINTERFACE_LIB
# define ENGINEINTERFACE_EXPORT Q_DECL_EXPORT
#else
# define ENGINEINTERFACE_EXPORT Q_DECL_IMPORT
#endif
#else
# define ENGINEINTERFACE_EXPORT
#endif
