#include <QtTest/QtTest>
#include "alientoken.h"

class TestAlienToken: public QObject
{
    Q_OBJECT
private:
    AlienToken* t;

private slots:
    void initTestCase()
    {
    }

    void create()
    {
        t = new AlienToken(100.0);
        QCOMPARE(t->energy, 100.0);
    }

    void cleanupTestCase()
    {
    }
};

//QTEST_MAIN(AlienTokenTest)
//#include "alientokentest.moc"
