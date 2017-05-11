#ifndef BASE_DEFINITIONS_H
#define BASE_DEFINITIONS_H

#include <QtGlobal>
#include <QVector2D>
#include <QVector2D>
#include <QSize>
#include <QMap>
#include <QRectF>
#include <QSet>
#include <QList>
#include <QDataStream>
#include <qmath.h>

#include <set>
#include <map>
#include <vector>
#include <unordered_set>
#include <unordered_map>

#include "blabla_global.h"

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
class BLABLA_EXPORT IntVector2D {
public:
	int x;
	int y;

	IntVector2D();
	IntVector2D(std::initializer_list<int> l);
	IntVector2D(QVector2D const& vec);
	QVector2D toQVector2D();
	IntVector2D& restrictToRect(IntRect const& rect);
	bool operator==(IntVector2D const& vec);
};

BLABLA_EXPORT std::ostream& operator << (std::ostream& os, const IntVector2D& vec);

class BLABLA_EXPORT IntRect {
public:
	IntVector2D p1;
	IntVector2D p2;

	IntRect() = default;
	IntRect(std::initializer_list<IntVector2D> l);
	IntRect(QRectF const& rect);
	bool isContained(IntVector2D const& p);
};


#endif // BASE_DEFINITIONS_H
