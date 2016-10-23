#ifndef TESTPHYSICS_H
#define TESTPHYSICS_H

#include <QtTest/QtTest>
#include "testsettings.h"

#include "model/physics/physics.h"
#include "global/globalfunctions.h"

class TestPhysics: public QObject
{
    Q_OBJECT
private slots:
    void initTestCase()
    {
    }

    void testRotateClockwise()
    {
        QVector3D v1 = QVector3D(0.0, -1.0, 0.0);
        QVector3D v2 = QVector3D(1.0, 0.0, 0.0);
        QVector3D v3 = QVector3D(0.0, 1.0, 0.0);
        QVector3D v4 = QVector3D(-1.0, 0.0, 0.0);
        QVector3D v1r = Physics::rotateClockwise(v1, 90.0);
        QVector3D v2r = Physics::rotateClockwise(v2, 90.0);
        QVector3D v3r = Physics::rotateClockwise(v3, 90.0);
        QVector3D v4r = Physics::rotateClockwise(v4, 90.0);
        QString s = QString("90 DEG rotation; vector before: (%1, %2), vector after: (%3, %4)").arg(v1.x()).arg(v1.y()).arg(v1r.x()).arg(v1r.y());
        QVERIFY2((v1r-v2).length() < TEST_REAL_PRECISION, s.toLatin1().data());
        s = QString("90 DEG rotation; vector before: (%1, %2), vector after: (%3, %4)").arg(v2.x()).arg(v2.y()).arg(v2r.x()).arg(v2r.y());
        QVERIFY2((v2r-v3).length() < TEST_REAL_PRECISION, s.toLatin1().data());
        s = QString("90 DEG rotation; vector before: (%1, %2), vector after: (%3, %4)").arg(v3.x()).arg(v3.y()).arg(v3r.x()).arg(v3r.y());
        QVERIFY2((v3r-v4).length() < TEST_REAL_PRECISION, s.toLatin1().data());
        s = QString("90 DEG rotation; vector before: (%1, %2), vector after: (%3, %4)").arg(v4.x()).arg(v4.y()).arg(v4r.x()).arg(v4r.y());
        QVERIFY2((v4r-v1).length() < TEST_REAL_PRECISION, s.toLatin1().data());
    }

    void testAngleOfVector()
    {
        qreal a = Physics::angleOfVector(QVector3D(0.0, -1.0, 0.0));
        QVERIFY(qAbs(a) < TEST_LOW_REAL_PRECISION);
        a = Physics::angleOfVector(QVector3D(1.0, 0.0, 0.0));
        QVERIFY(qAbs(a-90.0) < TEST_LOW_REAL_PRECISION);
        a = Physics::angleOfVector(QVector3D(0.0, 1.0, 0.0));
        QVERIFY(qAbs(a-180.0) < TEST_LOW_REAL_PRECISION);
        a = Physics::angleOfVector(QVector3D(-1.0, 0.0, 0.0));
        QVERIFY(qAbs(a-270.0) < TEST_LOW_REAL_PRECISION);
    }

    void testUnitVectorOfAngle ()
    {
        //test angle -> unit vector -> angle conversion
        for(int i = 0; i < 100; ++i) {
            qreal angleBefore = GlobalFunctions::random(0.0, 360.0);
            QVector3D v = Physics::unitVectorOfAngle(angleBefore);
            qreal angleAfter = Physics::angleOfVector(v);
            QString s = QString("angle before: %1, vector: (%2, %3), angle after: %4").arg(angleBefore).arg(v.x()).arg(v.y()).arg(angleAfter);
            QVERIFY2(qAbs(angleBefore-angleAfter) < TEST_LOW_REAL_PRECISION, s.toLatin1().data());
        }

        //test overrun
        for(qreal a = 0.0; a < 360.0; a += 10.0) {
            QVector3D v1 = Physics::unitVectorOfAngle(a);
            QVector3D v2 = Physics::unitVectorOfAngle(a+360.0);
            QVector3D v3 = Physics::unitVectorOfAngle(a-360.0);
            QString s = QString("vector1: (%1, %2), vector2: (%3, %4)").arg(v1.x()).arg(v1.y()).arg(v2.x()).arg(v2.y());
            QVERIFY2(qAbs((v1-v2).length()) < TEST_REAL_PRECISION, s.toLatin1().data());
            s = QString("vector2: (%1, %2), vector3: (%3, %4)").arg(v2.x()).arg(v2.y()).arg(v3.x()).arg(v3.y());
            QVERIFY2(qAbs((v2-v3).length()) < TEST_REAL_PRECISION, s.toLatin1().data());
        }
    }

    void testClockwiseAngleFromFirstToSecondVector ()
    {
        for(int i = 0; i < 100; ++i) {
            qreal angle = GlobalFunctions::random(0.0, 360.0);
            qreal angleIncrement = GlobalFunctions::random(-180.0, 180.0);
            QVector3D v1 = Physics::unitVectorOfAngle(angle);
            QVector3D v2 = Physics::unitVectorOfAngle(angle+angleIncrement);
            qreal returnedAngle = Physics::clockwiseAngleFromFirstToSecondVector(v1, v2);
            QString s = QString("vector1: (%1, %2), vector2: (%3, %4), angle increment: %5, returned angle: %6").arg(v1.x()).arg(v1.y()).arg(v2.x()).arg(v2.y()).arg(angleIncrement).arg(returnedAngle);
            QVERIFY2(Physics::compareEqualAngle(returnedAngle, angleIncrement, TEST_LOW_REAL_PRECISION), s.toLatin1().data());
        }
    }

    void cleanupTestCase()
    {
    }
};

//QTEST_MAIN(TestPhysics)
//#include "testphysics.moc"

#endif
