#ifndef PREDICATES_H
#define PREDICATES_H

#include <QVector3D>
#include <gtest/gtest.h>

#include "TestSettings.h"

bool predEqualMediumPrecision(qreal a, qreal b);
bool predEqualLowPrecision(qreal a, qreal b);
::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector3D a, QVector3D b);


#endif // PREDICATES_H
