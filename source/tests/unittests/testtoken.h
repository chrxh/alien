#ifndef TESTTOKEN_H
#define TESTTOKEN_H

#include <QtTest/QtTest>
#include "model/entities/token.h"

class TestToken: public QObject
{
    Q_OBJECT
private:
    Token* t;

private slots:
    void initTestCase()
    {
    }

    void testCreation()
    {
        t = new Token(100.0);
        QCOMPARE(t->energy, 100.0);
    }

    void cleanupTestCase()
    {
    }
};

//QTEST_MAIN(TestToken)
//#include "testalientoken.moc"

#endif
