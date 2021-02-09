#pragma once

#include <QVector2D>
#include <gtest/gtest.h>

#include "EngineInterface/Definitions.h"
#include "TestSettings.h"

bool predEqualIntVector(IntVector2D a, IntVector2D b);
bool predEqual(double a, double b, double precision);
bool predEqual_lowPrecision(double a, double b);
bool predEqual_mediumPrecision(qreal a, qreal b);
bool predEqual_relative(double a, double b);

bool predLessThan_MediumPrecision(double a, double b);
::testing::AssertionResult predEqualVectorMediumPrecision(const char* a_expr, const char* b_expr, QVector2D a, QVector2D b);

