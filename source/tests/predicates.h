#ifndef PREDICATES_H
#define PREDICATES_H

#include <QVector2D>
#include <gtest/gtest.h>

#include "Model/Api/Definitions.h"
#include "TestSettings.h"

bool predEqualIntVector(IntVector2D a, IntVector2D b);
bool predEqualMediumPrecision(qreal a, qreal b);
bool predEqualLowPrecision(qreal a, qreal b);
::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector2D a, QVector2D b);


#endif // PREDICATES_H
