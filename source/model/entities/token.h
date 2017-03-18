#ifndef TOKEN_H
#define TOKEN_H

#include <QVector>

#include "model/definitions.h"

class Cell;
class Token
{
public:
    Token (SimulationContext* context, qreal energy_ = 0.0, bool randomData = false);
    Token (SimulationContext* context, qreal energy_, QByteArray memory_);

    Token* duplicate ();
    int getTokenAccessNumber ();        //from memory[0]
    void setTokenAccessNumber (int i);

    void serializePrimitives (QDataStream& stream);
    void deserializePrimitives (QDataStream& stream);


	QByteArray memory;
    qreal energy;
//    Cell* linkStack[TOKEN_STACKSIZE];
//    int linkStackPointer;

private:
	SimulationContext* _context = nullptr;
};

#endif // TOKEN_H
