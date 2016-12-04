#include "predicates.h"

bool predEqualMediumPrecision(qreal a, qreal b)
{
	return qAbs(a - b) < TEST_MEDIUM_PRECISION;
}

::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector3D a, QVector3D b)
{
	if ((a - b).length() < TEST_MEDIUM_PRECISION)
		return ::testing::AssertionSuccess();
	else
		return ::testing::AssertionFailure() << a_expr << " = (" << a.x() << ", " << a.y() << ") and "
			<< b_expr << " = (" << b.x() << ", " << b.y() << ") do not coincide";
}

bool predEqualLowPrecision(qreal a, qreal b)
{
	return qAbs(a - b) < TEST_LOW_PRECISION;
}
