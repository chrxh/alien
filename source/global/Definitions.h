#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

#include <QtGlobal>
#include <QVector3D>
#include <QSize>
#include <QMap>
#include <QSet>
#include <qmath.h>
#include <QDataStream>

#include <set>
#include <unordered_set>

#define SET_CHILD(oldChild, newChild)\
	if (oldChild != newChild) { \
		delete oldChild; \
		oldChild = newChild; \
		oldChild->setParent(this); \
	}

#endif // GLOBAL_DEFINITIONS_H
