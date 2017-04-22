#include <gtest/gtest.h>

#include "tests/predicates.h"
#include "model/physics/physics.h"
#include "global/numbergenerator.h"

namespace {
	::testing::AssertionResult predVectorAfterRotation(const char* beforeExpr, const char* afterExpr, const char* expectedExpr
		, QVector3D before, QVector3D after, QVector3D expected)
	{
		::testing::AssertionResult result = predEqualVectorMediumPrecision(afterExpr, expectedExpr, after, expected);
		if (result)
			return result;
		else
			return ::testing::AssertionFailure() << "90 DEG rotation failed. Vector before: "
				<< beforeExpr << " = (" << before.x() << ", " << before.y() << "), vector after: "
				<< afterExpr << " = (" << after.x() << ", " << after.y() << "), vector expected: "
				<< expectedExpr << " = (" << expected.x() << ", " << expected.y() << ").";
	}

	::testing::AssertionResult predUnitVectorOfAngle(const char* beforeAngleExpr, const char* beforeVectorExpr, const char* afterAngleExpr
		, qreal beforeAngle, QVector3D beforeVector, qreal afterAngle)
	{
		if (Physics::compareEqualAngle(beforeAngle, afterAngle, TEST_LOW_PRECISION))
			return ::testing::AssertionSuccess();
		else
			return ::testing::AssertionFailure() << "angle-vector-conversion failed. Angle before: "
			<< beforeAngleExpr << " = " << beforeAngle << ", vector before: "
			<< beforeVectorExpr << " = (" << beforeVector.x() << ", " << beforeVector.y() << "), angle after: "
			<< afterAngleExpr << " = " << afterAngle << ".";
	}

	::testing::AssertionResult predRelativeAngleBetweenVectors(const char* firstVectorExpr, const char* secondVectorExpr, const char* calculatedRelativeAngleExpr
		, const char* expectedRelativeAngleExpr, QVector3D firstVector, QVector3D secondVector, qreal calculatedRelativeAngle, qreal expectedRelativeAngle)
	{
		if (Physics::compareEqualAngle(calculatedRelativeAngle, expectedRelativeAngle, TEST_LOW_PRECISION))
			return ::testing::AssertionSuccess();
		else
			return ::testing::AssertionFailure() << "relative angle between vectors is wrongly calculated. First vector: "
			<< firstVectorExpr << " = (" << firstVector.x() << ", " << firstVector.y() << "), second vector: "
			<< secondVectorExpr << " = (" << secondVector.x() << ", " << secondVector.y() << "), actual relative angle: "
			<< calculatedRelativeAngleExpr << " = " << calculatedRelativeAngle << ", expected relative angle: "
			<< expectedRelativeAngleExpr << " = " << expectedRelativeAngle << ".";
	}
}

TEST (TestPhysics, testRotateClockwise)
{
    QVector3D v1 = QVector3D(0.0, -1.0, 0.0);
    QVector3D v2 = QVector3D(1.0, 0.0, 0.0);
    QVector3D v3 = QVector3D(0.0, 1.0, 0.0);
    QVector3D v4 = QVector3D(-1.0, 0.0, 0.0);
    QVector3D v1r = Physics::rotateClockwise(v1, 90.0);
    QVector3D v2r = Physics::rotateClockwise(v2, 90.0);
    QVector3D v3r = Physics::rotateClockwise(v3, 90.0);
    QVector3D v4r = Physics::rotateClockwise(v4, 90.0);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v1, v1r, v2);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v2, v2r, v3);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v3, v3r, v4);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v4, v4r, v1);
}

TEST (TestPhysics, testAngleOfVector)
{
    qreal a = Physics::angleOfVector(QVector3D(0.0, -1.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, a, 0.0);
    a = Physics::angleOfVector(QVector3D(1.0, 0.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, a, 90.0);
    a = Physics::angleOfVector(QVector3D(0.0, 1.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, a, 180.0);
    a = Physics::angleOfVector(QVector3D(-1.0, 0.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, a, 270.0);
}

TEST (TestPhysics, testUnitVectorOfAngle)
{
    //test angle -> unit vector -> angle conversion
    for(int i = 0; i < 100; ++i) {
        qreal angleBefore = GlobalFunctions::random(0.0, 360.0);
        QVector3D v = Physics::unitVectorOfAngle(angleBefore);
        qreal angleAfter = Physics::angleOfVector(v);
		ASSERT_PRED_FORMAT3(predUnitVectorOfAngle, angleBefore, v, angleAfter);
    }

    //test overrun
    for(qreal a = 0.0; a < 360.0; a += 10.0) {
        QVector3D v1 = Physics::unitVectorOfAngle(a);
        QVector3D v2 = Physics::unitVectorOfAngle(a+360.0);
        QVector3D v3 = Physics::unitVectorOfAngle(a-360.0);
		ASSERT_PRED_FORMAT2(predEqualVectorMediumPrecision, v1, v2);
		ASSERT_PRED_FORMAT2(predEqualVectorMediumPrecision, v2, v3);
    }
}

TEST (TestPhysics, testClockwiseAngleFromFirstToSecondVector)
{
    for(int i = 0; i < 100; ++i) {
        qreal angle = GlobalFunctions::random(0.0, 360.0);
        qreal angleIncrement = GlobalFunctions::random(-180.0, 180.0);
        QVector3D v1 = Physics::unitVectorOfAngle(angle);
        QVector3D v2 = Physics::unitVectorOfAngle(angle+angleIncrement);
        qreal returnedAngle = Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
		ASSERT_PRED_FORMAT4(predRelativeAngleBetweenVectors, v1, v2, returnedAngle, angleIncrement);
    }
}

