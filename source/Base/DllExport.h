#ifndef BASE_DLLEXPORT_H
#define BASE_DLLEXPORT_H

#include <QtCore/qglobal.h>

#ifndef ALIEN_STATIC
	#ifdef BASE_LIB
	# define BASE_EXPORT Q_DECL_EXPORT
	#else
	# define BASE_EXPORT Q_DECL_IMPORT
	#endif
#else
	# define BASE_EXPORT
#endif

#endif // BASE_DLLEXPORT_H
