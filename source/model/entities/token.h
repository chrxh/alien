#ifndef TOKEN_H
#define TOKEN_H

#include <QVector>

class Cell;
class Token
{
public:
    Token (qreal energy_ = 0.0, bool randomData = false);
    Token (qreal energy_, QVector< quint8 > memory_);
    Token (QDataStream& stream);

    Token* duplicate ();
    int getTokenAccessNumber ();        //from memory[0]
    void setTokenAccessNumber (int i);

    void serialize (QDataStream& stream);


    QVector< quint8 >memory;
    qreal energy;
//    Cell* linkStack[TOKEN_STACKSIZE];
//    int linkStackPointer;
};

#endif // TOKEN_H
