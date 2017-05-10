#ifndef GLOBAL_DEFINITIONS_H
#define GLOBAL_DEFINITIONS_H

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

struct IntVector2D {
	int x;
	int y;

	IntVector2D() : x(0), y(0) { }
	IntVector2D(std::initializer_list<int> l)
	{
		auto it = l.begin();
		x = *it++;
		y = *it;
	}
	IntVector2D(QVector2D const& vec) : x(static_cast<int>(vec.x())), y(static_cast<int>(vec.y())) { }
	QVector2D toQVector2D() { return QVector2D(x, y); }
};

extern bool operator==(IntVector2D const& vec1, IntVector2D const& vec2);
extern std::ostream& operator << (std::ostream& os, const IntVector2D& vec);

struct IntRect {
	IntVector2D p1;
	IntVector2D p2;

	bool isContained(IntVector2D p)
	{
		return p1.x <= p.x && p1.y <= p.y && p.x <= p2.x && p.y <= p2.y;
	}
};


#define SET_CHILD(previousChild, newChild)\
	if (previousChild != newChild) { \
		delete previousChild; \
		previousChild = newChild; \
		previousChild->setParent(this); \
	}

#endif // GLOBAL_DEFINITIONS_H
