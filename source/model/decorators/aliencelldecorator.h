#ifndef ALIENCELLDECORATOR_H
#define ALIENCELLDECORATOR_H

#include "model/entities/aliencell.h"

class AlienCellDecorator : public AlienCell
{
public:
    AlienCellDecorator (AlienCell* cell) : _cell(cell) {}
    virtual ~AlienCellDecorator () {}

    template< typename T >
    T* extractObject (AlienCell* cell);

protected:
    AlienCell* _cell;
};


template< typename T >
T* AlienCellDecorator::extractObject (AlienCell* cell)
{
    T* object = dynamic_cast< T* >(cell);
    if( object )
        return object;
    else {
        AlienCellDecorator* decorator = dynamic_cast< AlienCellDecorator* >(cell);
        if( decorator )
            return AlienCellDecorator::extractObject<T>(decorator->_cell);
        else
            return 0;
    }
}

#endif // ALIENCELLDECORATOR_H
