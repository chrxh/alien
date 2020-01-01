#pragma once

#include <QVector2D>
#include <gtest/gtest.h>

#include "ModelBasic/Definitions.h"
#include "TestSettings.h"

bool predEqualIntVector(IntVector2D a, IntVector2D b);
bool predEqualMediumPrecision(qreal a, qreal b);
bool predLessThanMediumPrecision(double a, double b);
bool predEqualLowPrecision(double a, double b);
bool predEqual(double a, double b, double precision);
::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector2D a, QVector2D b);

