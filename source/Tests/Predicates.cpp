#include "Predicates.h"

bool predEqualIntVector(IntVector2D a, IntVector2D b)
{
	return a == b;
}

bool predEqualMediumPrecision(double a, double b)
{
	return qAbs(a - b) < FLOATINGPOINT_MEDIUM_PRECISION;
}

bool predLessThanMediumPrecision(double a, double b)
{
	return a <= b + FLOATINGPOINT_MEDIUM_PRECISION;
}

bool predEqualLowPrecision(double a, double b)
{
	return qAbs(a - b) < FLOATINGPOINT_LOW_PRECISION;
}

bool predEqual(double a, double b, double precision)
{
    return qAbs(a - b) < precision;
}

::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector2D a, QVector2D b)
{
	if ((a - b).length() < FLOATINGPOINT_MEDIUM_PRECISION)
		return ::testing::AssertionSuccess();
	else
		return ::testing::AssertionFailure() << a_expr << " = (" << a.x() << ", " << a.y() << ") and "
			<< b_expr << " = (" << b.x() << ", " << b.y() << ") do not coincide";
}

