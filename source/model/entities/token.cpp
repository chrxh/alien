#include "model/config.h"
#include "model/simulationparameters.h"
#include "model/simulationcontext.h"

#include "token.h"

Token::Token(SimulationContext* context, qreal energy_, bool randomData)
    : energy(energy_), _context(context)
{
	memory = QByteArray(context->getSimulationParameters()->TOKEN_MEMSIZE, 0);
	if (randomData) {
        for( int i = 0; i < context->getSimulationParameters()->TOKEN_MEMSIZE; ++i )
            memory[i] = qrand()%256;
    }
}


Token::Token (SimulationContext* context, qreal energy_, QByteArray memory_)
    : energy(energy_), _context(context)
{
	memory = memory_.left(context->getSimulationParameters()->TOKEN_MEMSIZE);
}

Token* Token::duplicate ()
{
    Token* newToken(new Token(_context));
    for( int i = 0; i < _context->getSimulationParameters()->TOKEN_MEMSIZE; ++i )
        newToken->memory[i] = memory[i];
    newToken->energy = energy;
/*    newToken->linkStackPointer = linkStackPointer;
    for( int i = 0; i < linkStackPointer; ++i )
        newToken->linkStack[i] = linkStack[i];*/

    return newToken;
}

int Token::getTokenAccessNumber ()
{
    return memory[0]% _context->getSimulationParameters()->MAX_TOKEN_ACCESS_NUMBERS;
}

void Token::setTokenAccessNumber (int i)
{
    memory[0] = i;
}

void Token::serializePrimitives (QDataStream& stream)
{
    stream << memory << energy;
}

void Token::deserializePrimitives(QDataStream& stream)
{
	stream >> memory >> energy;
	auto memSize = _context->getSimulationParameters()->TOKEN_MEMSIZE;
	memory = memory.left(memSize);
	memory.resize(memSize);
}
