#include "token.h"

#include "model/config.h"

Token::Token(qreal energy_, bool randomData)
    : memory(simulationParameters.TOKEN_MEMSIZE, 0), energy(energy_)
{
    if( randomData ) {
        for( int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
            memory[i] = qrand()%256;
    }
}


Token::Token (qreal energy_, QByteArray memory_)
    : energy(energy_)
{
	memory = memory_.left(simulationParameters.TOKEN_MEMSIZE);
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

void Token::serializePrimitives (QDataStream& stream)
{
    stream << simulationParameters.TOKEN_MEMSIZE;
    for(int i = 0; i < simulationParameters.TOKEN_MEMSIZE; ++i )
        stream << memory[i];
    stream << energy;
}

void Token::deserializePrimitives (QDataStream& stream)
{
    memory.resize(simulationParameters.TOKEN_MEMSIZE);

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
