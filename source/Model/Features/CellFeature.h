#ifndef CELLFEATURE_H
#define CELLFEATURE_H

#include <QDataStream>

#include "Model/Definitions.h"

class CellFeature
{
public:
    CellFeature (UnitContext* context) : _context(context) {}
    virtual ~CellFeature ();

	virtual void setContext(UnitContext* context);

    void registerNextFeature (CellFeature* nextFeature);
    struct ProcessingResult {
        bool decompose;
        EnergyParticle* newEnergyParticle;
    };
    ProcessingResult process (Token* token, Cell* cell, Cell* previousCell);
	void mutate();

	virtual void serializePrimitives(QDataStream& stream) const {}
	virtual void deserializePrimitives(QDataStream& stream) {}

protected:
    virtual ProcessingResult processImpl (Token* token, Cell* cell, Cell* previousCell) = 0;
	virtual void mutateImpl() {};

    UnitContext* _context = nullptr;
    CellFeature* _nextFeature = nullptr;

public:
    template< typename T >
    inline T* findObject ();
};

/******************** inline methods ******************/
template< typename T >
T* CellFeature::findObject ()
{
    T* object = dynamic_cast< T* >(this);
    if( object )
        return object;
    else {
        if( _nextFeature )
            return _nextFeature->findObject<T>();
        else
            return 0;
    }
}

#endif // CELLFEATURE_H
