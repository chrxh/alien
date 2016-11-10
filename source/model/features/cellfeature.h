#ifndef CELLDECORATOR_H
#define CELLDECORATOR_H

#include <QDataStream>

class Grid;
class Cell;
class Token;
class EnergyParticle;

class CellDecorator
{
public:
    CellDecorator (Grid*& grid) : _grid(grid), _nextFeature(0) {}
    virtual ~CellDecorator ();

    void registerNextFeature (CellDecorator* nextFeature);
    struct ProcessingResult {
        bool decompose;
        EnergyParticle* newEnergyParticle;
    };
    ProcessingResult process (Token* token, Cell* cell, Cell* previousCell);
    void serialize (QDataStream& stream) const;

protected:
    virtual ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) = 0;
    virtual void serializeImpl (QDataStream& stream) const {}

    Grid* _grid;
    CellDecorator* _nextFeature;

public:
    template< typename T >
    static T* findObject (CellDecorator* feature);
};


template< typename T >
T* CellDecorator::findObject (CellDecorator* feature)
{
    T* object = dynamic_cast< T* >(feature);
    if( object )
        return object;
    else {
        if( feature->_nextFeature )
            return CellDecorator::findObject<T>(feature->_nextFeature);
        else
            return 0;
    }
}

#endif // CELLDECORATOR_H
