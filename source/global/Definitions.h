#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

#include <QtGlobal>
#include <QVector3D>
#include <QVector2D>
#include <QSize>
#include <QMap>
#include <QSet>
#include <QList>
#include <QDataStream>
#include <qmath.h>

#include <set>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>

class NumberGenerator;
class TagGenerator;

using std::uint64_t;
using std::uint32_t;
using std::uint32_t;
using std::int64_t;
using std::int32_t;
using std::int32_t;
using std::vector;
using std::map;
using std::set;
using std::unordered_set;
using std::unordered_map;
using std::pair;

#define SET_CHILD(oldChild, newChild)\
	if (oldChild != newChild) { \
		delete oldChild; \
		oldChild = newChild; \
		oldChild->setParent(this); \
	}

#endif // GLOBAL_DEFINITIONS_H
