#include "predicates.h"

bool predEqualIntVector(IntVector2D a, IntVector2D b)
{
	return a == b;
}

bool predEqualMediumPrecision(qreal a, qreal b)
{
	return qAbs(a - b) < FLOATINGPOINT_MEDIUM_PRECISION;
}

bool predEqualLowPrecision(qreal a, qreal b)
{
	return qAbs(a - b) < FLOATINGPOINT_LOW_PRECISION;
}

::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector3D a, QVector3D b)
{
	if ((a - b).length() < FLOATINGPOINT_MEDIUM_PRECISION)
		return ::testing::AssertionSuccess();
	else
		return ::testing::AssertionFailure() << a_expr << " = (" << a.x() << ", " << a.y() << ") and "
			<< b_expr << " = (" << b.x() << ", " << b.y() << ") do not coincide";
}

