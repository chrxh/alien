#ifndef BASE_DEFINITIONS_H
#define BASE_DEFINITIONS_H

#include <QtGlobal>
#include <QVector2D>
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

#include "DllExport.h"
#include "Tracker.h"

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
using std::list;
using std::unordered_set;
using std::unordered_map;
using std::pair;

class IntRect;
struct BASE_EXPORT IntVector2D {
	int x;
	int y;

	IntVector2D();
	IntVector2D(std::initializer_list<int> l);
	IntVector2D(QVector2D const& vec);
	QVector2D toQVector2D();
	IntVector2D& restrictToRect(IntRect const& rect);
	bool operator==(IntVector2D const& vec);
};

BASE_EXPORT std::ostream& operator << (std::ostream& os, const IntVector2D& vec);

struct BASE_EXPORT IntRect {
	IntVector2D p1;
	IntVector2D p2;

	IntRect() = default;
	IntRect(std::initializer_list<IntVector2D> l);
	IntRect(QRectF const& rect);
	bool isContained(IntVector2D const& p);
};


#define SET_CHILD(previousChild, newChild)\
	if (previousChild != newChild) { \
		delete previousChild; \
		previousChild = newChild; \
		previousChild->setParent(this); \
	}

#endif // BASE_DEFINITIONS_H
