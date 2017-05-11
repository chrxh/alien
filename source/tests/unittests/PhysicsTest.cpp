#include <gtest/gtest.h>

#include "Base/ServiceLocator.h"
#include "Base/GlobalFactory.h"
#include "Base/NumberGenerator.h"
#include "model/Physics/Physics.h"
#include "tests/predicates.h"

class PhysicsTest : public ::testing::Test
{
public:
	PhysicsTest();
	~PhysicsTest();

protected:
	NumberGenerator* _numberGen = nullptr;
};

PhysicsTest::PhysicsTest()
{
	GlobalFactory* factory = ServiceLocator::getInstance().getService<GlobalFactory>();
	_numberGen = factory->buildRandomNumberGenerator();
	_numberGen->init(123123, 0);
}

PhysicsTest::~PhysicsTest()
{
	delete _numberGen;
}

namespace {
	::testing::AssertionResult predVectorAfterRotation(const char* beforeExpr, const char* afterExpr, const char* expectedExpr
		, QVector2D before, QVector2D after, QVector2D expected)
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
		, qreal beforeAngle, QVector2D beforeVector, qreal afterAngle)
	{
		if (Physics::compareEqualAngle(beforeAngle, afterAngle, FLOATINGPOINT_LOW_PRECISION))
			return ::testing::AssertionSuccess();
		else
			return ::testing::AssertionFailure() << "angle-vector-conversion failed. Angle before: "
			<< beforeAngleExpr << " = " << beforeAngle << ", vector before: "
			<< beforeVectorExpr << " = (" << beforeVector.x() << ", " << beforeVector.y() << "), angle after: "
			<< afterAngleExpr << " = " << afterAngle << ".";
	}

	::testing::AssertionResult predRelativeAngleBetweenVectors(const char* firstVectorExpr, const char* secondVectorExpr, const char* calculatedRelativeAngleExpr
		, const char* expectedRelativeAngleExpr, QVector2D firstVector, QVector2D secondVector, qreal calculatedRelativeAngle, qreal expectedRelativeAngle)
	{
		if (Physics::compareEqualAngle(calculatedRelativeAngle, expectedRelativeAngle, FLOATINGPOINT_LOW_PRECISION))
			return ::testing::AssertionSuccess();
		else
			return ::testing::AssertionFailure() << "relative angle between vectors is wrongly calculated. First vector: "
			<< firstVectorExpr << " = (" << firstVector.x() << ", " << firstVector.y() << "), second vector: "
			<< secondVectorExpr << " = (" << secondVector.x() << ", " << secondVector.y() << "), actual relative angle: "
			<< calculatedRelativeAngleExpr << " = " << calculatedRelativeAngle << ", expected relative angle: "
			<< expectedRelativeAngleExpr << " = " << expectedRelativeAngle << ".";
	}
}

TEST_F (PhysicsTest, testRotateClockwise)
{
    QVector2D v1 = QVector2D(0.0, -1.0);
    QVector2D v2 = QVector2D(1.0, 0.0);
    QVector2D v3 = QVector2D(0.0, 1.0);
    QVector2D v4 = QVector2D(-1.0, 0.0);
    QVector2D v1r = Physics::rotateClockwise(v1, 90.0);
    QVector2D v2r = Physics::rotateClockwise(v2, 90.0);
    QVector2D v3r = Physics::rotateClockwise(v3, 90.0);
    QVector2D v4r = Physics::rotateClockwise(v4, 90.0);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v1, v1r, v2);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v2, v2r, v3);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v3, v3r, v4);
	ASSERT_PRED_FORMAT3(predVectorAfterRotation, v4, v4r, v1);
}

TEST_F(PhysicsTest, testAngleOfVector)
{
    qreal a = Physics::angleOfVector(QVector2D(0.0, -1.0));
	ASSERT_PRED2(predEqualLowPrecision, 0.0, a);
    a = Physics::angleOfVector(QVector2D(1.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, 90.0, a);
    a = Physics::angleOfVector(QVector2D(0.0, 1.0));
	ASSERT_PRED2(predEqualLowPrecision, 180.0, a);
    a = Physics::angleOfVector(QVector2D(-1.0, 0.0));
	ASSERT_PRED2(predEqualLowPrecision, 270.0, a);
}

TEST_F(PhysicsTest, testUnitVectorOfAngle)
{
    //test angle -> unit vector -> angle conversion
    for(int i = 0; i < 100; ++i) {
        qreal angleBefore = _numberGen->getRandomReal(0.0, 360.0);
        QVector2D v = Physics::unitVectorOfAngle(angleBefore);
        qreal angleAfter = Physics::angleOfVector(v);
		ASSERT_PRED_FORMAT3(predUnitVectorOfAngle, angleBefore, v, angleAfter);
    }

    //test overrun
    for(qreal a = 0.0; a < 360.0; a += 10.0) {
        QVector2D v1 = Physics::unitVectorOfAngle(a);
        QVector2D v2 = Physics::unitVectorOfAngle(a+360.0);
        QVector2D v3 = Physics::unitVectorOfAngle(a-360.0);
		ASSERT_PRED_FORMAT2(predEqualVectorMediumPrecision, v1, v2);
		ASSERT_PRED_FORMAT2(predEqualVectorMediumPrecision, v2, v3);
    }
}

TEST_F(PhysicsTest, testClockwiseAngleFromFirstToSecondVector)
{
    for(int i = 0; i < 100; ++i) {
        qreal angle = _numberGen->getRandomReal(0.0, 360.0);
        qreal angleIncrement = _numberGen->getRandomReal(-180.0, 180.0);
        QVector2D v1 = Physics::unitVectorOfAngle(angle);
        QVector2D v2 = Physics::unitVectorOfAngle(angle+angleIncrement);
        qreal returnedAngle = Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
		ASSERT_PRED_FORMAT4(predRelativeAngleBetweenVectors, v1, v2, returnedAngle, angleIncrement);
    }
}

