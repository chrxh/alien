#include "Base/NumberGenerator.h"
#include "ModelBasic/Settings.h"
#include "ModelBasic/SimulationParameters.h"
#include "UnitContext.h"

#include "TokenImpl.h"

TokenImpl::TokenImpl(UnitContext * context)
	: _context(context), _memory(context->getSimulationParameters().tokenMemorySize, 0)
{
}

TokenImpl::TokenImpl(UnitContext* context, qreal energy, QByteArray const& memory)
	: _context(context), _energy(energy)
{

	int memorySize = context->getSimulationParameters().tokenMemorySize;
	_memory = memory.left(memorySize);
	if (memorySize > _memory.size()) {
		_memory.append(memorySize - _memory.size(), 0);
	}
}

void TokenImpl::setContext(UnitContext * context)
{
	_context = context;
}

TokenImpl* TokenImpl::duplicate() const
{
	TokenImpl* newToken(new TokenImpl(_context));
	for (int i = 0; i < _context->getSimulationParameters().tokenMemorySize; ++i)
		newToken->_memory[i] = _memory[i];
	newToken->_energy = _energy;

	return newToken;
}

TokenDescription TokenImpl::getDescription() const
{
	return TokenDescription().setEnergy(_energy).setData(_memory);
}

int TokenImpl::getTokenAccessNumber() const
{
	return static_cast<unsigned char>(_memory[0]) % _context->getSimulationParameters().cellMaxTokenBranchNumber;
}

void TokenImpl::setTokenAccessNumber(int i)
{
	_memory[0] = i;
}

void TokenImpl::setEnergy(qreal energy)
{
	_energy = energy;
}

qreal TokenImpl::getEnergy() const
{
	return _energy;
}

QByteArray & TokenImpl::getMemoryRef() 
{
	return _memory;
}

