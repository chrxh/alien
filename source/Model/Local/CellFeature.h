#pragma once

#include <QDataStream>

#include "Model/Api/Definitions.h"
#include "Model/Api/Descriptions.h"

class CellFeature
{
public:
    CellFeature (UnitContext* context) : _context(context) {}
    virtual ~CellFeature ();

	virtual void setContext(UnitContext* context);

	virtual CellFeatureDescription getDescription() const;

    void registerNextFeature (CellFeature* nextFeature);
    struct ProcessingResult {
        bool decompose;
        Particle* newEnergyParticle;
    };
    ProcessingResult process (Token* token, Cell* cell, Cell* previousCell);
	void mutate();

	virtual void serializePrimitives(QDataStream& stream) const {}
	virtual void deserializePrimitives(QDataStream& stream) {}

protected:
	virtual void getDescriptionImpl(CellFeatureDescription& desc) const {};
	virtual ProcessingResult processImpl(Token* token, Cell* cell, Cell* previousCell) = 0;
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

