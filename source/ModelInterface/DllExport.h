#pragma once

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
#ifdef MODELINTERFACE_LIB
# define MODELINTERFACE_EXPORT Q_DECL_EXPORT
#else
# define MODELINTERFACE_EXPORT Q_DECL_IMPORT
#endif
#else
# define MODELINTERFACE_EXPORT
#endif
