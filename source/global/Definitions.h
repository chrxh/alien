#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

#include <QtGlobal>
#include <QVector3D>
#include <QSize>
#include <QMap>
#include <QSet>
#include <QList>
#include <QDataStream>
#include <qmath.h>

#include <set>
#include <unordered_set>

class RandomNumberGenerator;
class TagGenerator;

#define SET_CHILD(oldChild, newChild)\
	if (oldChild != newChild) { \
		delete oldChild; \
		oldChild = newChild; \
		oldChild->setParent(this); \
	}

#endif // GLOBAL_DEFINITIONS_H
