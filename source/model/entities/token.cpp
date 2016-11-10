#include "token.h"

#include "model/simulationsettings.h"

Token::Token(qreal energy_, bool randomData)
    : memory(simulationParameters.TOKEN_MEMSIZE), energy(energy_)
{
    if( randomData ) {
        for( int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
            memory[i] = qrand()%256;
    }
    else {
        for( int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
            memory[i] = 0;
    }
}

Token::Token (QDataStream& stream)
: memory(simulationParameters.TOKEN_MEMSIZE)
{
    int memSize;
    stream >> memSize;
    quint8 data;
    for(int i = 0; i < memSize; ++i ) {
        if( i < simulationParameters.TOKEN_MEMSIZE ) {
            stream >> data;
            memory[i] = data;
        }
        else {
            stream >> data;
        }
    }
    for(int i = memSize; i < simulationParameters.TOKEN_MEMSIZE; ++i)
        memory[i] = 0;
    stream >> energy;
}

Token::Token (qreal energy_, QVector< quint8 > memory_)
    : memory(simulationParameters.TOKEN_MEMSIZE), energy(energy_)
{
    for( int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
        memory[i] = memory_[i];
}

Token* Token::duplicate ()
{
    Token* newToken(new Token());
    for( int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
        newToken->memory[i] = memory[i];
    newToken->energy = energy;
/*    newToken->linkStackPointer = linkStackPointer;
    for( int i = 0; i < linkStackPointer; ++i )
        newToken->linkStack[i] = linkStack[i];*/

    return newToken;
}

int Token::getTokenAccessNumber ()
{
    return memory[0]%simulationParameters.MAX_TOKEN_ACCESS_NUMBERS;
}

void Token::setTokenAccessNumber (int i)
{
    memory[0] = i;
}

void Token::serialize (QDataStream& stream)
{
    stream << simulationParameters.TOKEN_MEMSIZE;
    for(int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
        stream << memory[i];
    stream << energy;
}
/*
#include <QtTest/QtTest>

class TestToken: public QObject
{
    Q_OBJECT
private slots:
    void test1()
    {
        QString str = "Hello";
        QCOMPARE(str.toUpper(), QString("HELLO"));
    }
};

QTEST_MAIN(TestToken)
#include "alientoken.moc"
*/
