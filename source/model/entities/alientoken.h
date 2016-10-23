#ifndef ALIENTOKEN_H
#define ALIENTOKEN_H

#include <QVector>

class AlienCell;
class AlienToken
{
public:
    AlienToken (qreal energy_ = 0.0, bool randomData = false);
    AlienToken (qreal energy_, QVector< quint8 > memory_);
    AlienToken (QDataStream& stream);

    AlienToken* duplicate ();
    int getTokenAccessNumber ();        //from memory[0]
    void setTokenAccessNumber (int i);

    void serialize (QDataStream& stream);


    QVector< quint8 >memory;
    qreal energy;
//    AlienCell* linkStack[TOKEN_STACKSIZE];
//    int linkStackPointer;
};

#endif // ALIENTOKEN_H
